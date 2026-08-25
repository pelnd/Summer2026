# event_count

import time

import numpy as np
import torch
import cv2

from spikingjelly.activation_based import functional, surrogate, neuron
from spikingjelly.activation_based.model import parametric_lif_net
from metavision_core.event_io import EventsIterator


DEVICE = 'cpu'
T = 8
CHANNELS = 64
NUM_CLASSES = 11
H, W = 128, 128
MODEL_PATH = './checkpoint_latest.pth'
CLASS_NAMES = [
    'hand clap', 'right hand wave', 'other gestures',
    'left hand wave', 'right arm clockwise',
    'right arm counter clockwise', 'left arm clockwise',
    'left arm counter clockwise', 'arm rolls',
    'air drums', 'air guitar',
]

EVENTS_PER_FRAME = 50000   # events per model frame -- must match fine-tuning windowing exactly

# small, frequent batches from the camera -- purely for display refresh cadence,
# independent of EVENTS_PER_FRAME. Same idea as record.py's DISPLAY_N_EVENTS split:
# the model still only sees exact EVENTS_PER_FRAME-sized chunks (built by accumulating
# these small batches in main()), this just controls how often the window redraws.
DISPLAY_N_EVENTS = 5000
DISPLAY_MAX_WAIT_US = 100_000

# if a model frame hasn't accumulated EVENTS_PER_FRAME within this many seconds
# (subject not doing anything), discard the partial frame and reset instead of
# waiting indefinitely -- tune if it feels too trigger-happy or too sluggish live.
IDLE_TIMEOUT_S = 2.0

NATIVE_H, NATIVE_W = 320, 320

# BGR (OpenCV order) -- blue-family palette instead of red/green, since red+green
# overlap combines into a harsh yellow wherever both polarities fire in the same
# window. Blending two blues stays in-family instead of clashing.
POS_COLOR = np.array([250, 206, 135], dtype=np.float32)   # light blue, positive polarity
NEG_COLOR = np.array([112, 25, 25], dtype=np.float32)      # dark blue, negative polarity

def events_to_frame_cropped(events):
    #('x','y','p','t')

    x = (events['x'].astype(np.int64) * W) // NATIVE_W
    y = (events['y'].astype(np.int64) * H) // NATIVE_H
    p = events['p'].astype(np.int64)

    # normalize polarity to {0, 1} in case this sensor/SDK version uses {-1, 1}
    if p.size > 0 and p.min() < 0:
        p = (p > 0).astype(np.int64)

    frame = np.zeros((2, H, W), dtype=np.float32)
    np.add.at(frame, (p, y, x), 1.0)
    return frame

def genx320_camera():

    mv_iterator = EventsIterator("", mode="mixed", n_events=DISPLAY_N_EVENTS, delta_t=DISPLAY_MAX_WAIT_US)
    device = mv_iterator.reader.device
    biases = device.get_i_ll_biases()
    biases.set("bias_diff_on", 25)
    biases.set("bias_diff_off", 28)
    print(f'bias readback: on={biases.get("bias_diff_on")}  off={biases.get("bias_diff_off")}')

    height, width = mv_iterator.get_size()
    print(f'camera opened, reported resolution: {width}x{height}')


    assert (height, width) == (NATIVE_H, NATIVE_W), (
        f'expected {NATIVE_H}x{NATIVE_W}, got {height}x{width} -- '
        f'crop offsets would be wrong, fix NATIVE_H/NATIVE_W before continuing'
    )

    for events in mv_iterator:
        # NOTE: empty batches are yielded too (not skipped) -- main() needs these
        # to tick roughly every DISPLAY_MAX_WAIT_US even during true silence, so
        # the idle timeout can actually fire.
        yield events



def main():
    net = parametric_lif_net.DVSGestureNet(
        channels=CHANNELS,
        spiking_neuron=neuron.LIFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
    )

    functional.set_step_mode(net, 'm')
    net.to(DEVICE)
    net.eval()

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    net.load_state_dict(checkpoint['net'])
    print(f'loaded weights from {MODEL_PATH}')

    frame_buffer = []
    pred = -1
    cv2.namedWindow("SNN Prediction", cv2.WINDOW_NORMAL)
    spans = []

    accum = []        # small display batches, accumulated toward one EVENTS_PER_FRAME model frame
    accum_count = 0
    accum_deadline = None   # wall-clock deadline (time.perf_counter()) for the current partial frame

    for events in genx320_camera():
        if accum_deadline is None:
            accum_deadline = time.perf_counter() + IDLE_TIMEOUT_S

        accum.append(events)
        accum_count += events.size

        # --- display: redraws every small batch, independent of the model's window size.
        # shows only THIS tick's events (not the running accum toward EVENTS_PER_FRAME) --
        # using the growing accumulation instead made the image visibly snap back to
        # near-empty every time a model frame completed, looking like a flicker/strobe.
        disp_frame = events_to_frame_cropped(events)

        # scale bumped up ~10x vs. the old single-batch display (was sized up to
        # EVENTS_PER_FRAME=50000 per tick, now DISPLAY_N_EVENTS=5000) -- retune visually
        # if it looks too dim/bright.
        pos_mag = np.clip(disp_frame[1] * 200, 0, 255)   # positive polarity intensity, 0..255
        neg_mag = np.clip(disp_frame[0] * 200, 0, 255)   # negative polarity intensity, 0..255

        event_display = (pos_mag[..., None] / 255.0 * POS_COLOR
                          + neg_mag[..., None] / 255.0 * NEG_COLOR)
        event_display = np.clip(event_display, 0, 255).astype(np.uint8)
        event_display = cv2.resize(event_display, (512, 512))

        cv2.putText(
            event_display,
            f"Class: {CLASS_NAMES[pred] if pred >= 0 else '...'}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            3
        )
        cv2.imshow("SNN Prediction", event_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # --- idle timeout: subject not doing anything, discard the partial frame ---
        if accum_count < EVENTS_PER_FRAME and time.perf_counter() > accum_deadline:
            print('idle timeout -- no gesture detected, resetting')
            accum = []
            accum_count = 0
            accum_deadline = None
            frame_buffer = []   # a long real-time gap mid-window would stitch together
            pred = -1           # temporally disjoint frames the model never saw in training
            continue

        # --- model: only build a frame once exactly EVENTS_PER_FRAME events have accumulated ---
        if accum_count < EVENTS_PER_FRAME:
            continue

        all_accum = np.concatenate(accum)
        frame_events = all_accum[:EVENTS_PER_FRAME]   # exact match to training windowing
        leftover = all_accum[EVENTS_PER_FRAME:]        # carried into the next frame, not discarded

        frame = events_to_frame_cropped(frame_events)
        spans.append((frame_events['t'][-1] - frame_events['t'][0]) / 1000.0)
        frame_buffer.append(frame)

        accum = [leftover] if leftover.size > 0 else []
        accum_count = leftover.size
        accum_deadline = None   # restart the idle clock for the next frame's accumulation

        if len(frame_buffer) == T:

            occ = np.mean([(f > 0).mean() for f in frame_buffer])
            tot = np.mean([f.sum()  for f in frame_buffer])
            print(f'e/f: {tot:.0f}  non-zero: {occ:.4f}')

            x_in = np.stack(frame_buffer)[:, None]           # [T, 1, 2, H, W]
            x_in = torch.from_numpy(x_in).float().to(DEVICE)

            with torch.no_grad():
                out_firing_rate = net(x_in).mean(0)
                pred = out_firing_rate.argmax(1).item()

            functional.reset_net(net)

            print(f'predicted class: {pred}  '
                  f'(raw firing rates: {out_firing_rate[0].tolist()})')

            print(f'frame spans: {[f"{s:.0f}" for s in spans]}')
            spans = []
            frame_buffer = []

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
