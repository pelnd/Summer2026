# End-to-end LIVE pipeline latency: event capture + windowing + inference, on the Pi.
#
# Reuses app3.py's exact camera/windowing logic (same DISPLAY_N_EVENTS split for
# accumulation granularity, same exact-EVENTS_PER_FRAME slicing with carryover) but
# strips the display/idle-timeout code and adds timing instead. Measures REAL
# prediction-to-prediction latency during live use -- as opposed to
# measure_latency_pi.py, which times only the forward pass on pre-recorded clips
# (no camera, no event-accumulation wait).
#
# Keep gesturing continuously while this runs -- idle gaps get folded into the
# latency numbers as genuine (very large) outliers, same as they would in real use.

import csv
import time

import numpy as np
import torch

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

EVENTS_PER_FRAME = 50000        # must match fine-tuning windowing exactly
DISPLAY_N_EVENTS = 5000         # accumulation granularity only, same as app3.py -- no display here
DISPLAY_MAX_WAIT_US = 100_000

NATIVE_H, NATIVE_W = 320, 320

N_PREDICTIONS = 30              # how many predictions to collect before stopping -- edit as needed
OUT_CSV = './pipeline_latency_pi.csv'


def events_to_frame_cropped(events):
    x = (events['x'].astype(np.int64) * W) // NATIVE_W
    y = (events['y'].astype(np.int64) * H) // NATIVE_H
    p = events['p'].astype(np.int64)
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
        yield events


def main():
    print(f'device={DEVICE}  torch CPU threads={torch.get_num_threads()}')

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
    accum = []
    accum_count = 0

    prediction_latencies_ms = []
    rows = []
    last_pred_t = None   # wall-clock time of the previous completed prediction

    print(f'collecting {N_PREDICTIONS} predictions -- keep gesturing continuously for a clean measurement')

    for events in genx320_camera():
        accum.append(events)
        accum_count += events.size

        if accum_count < EVENTS_PER_FRAME:
            continue

        all_accum = np.concatenate(accum)
        frame_events = all_accum[:EVENTS_PER_FRAME]   # exact match to training windowing
        leftover = all_accum[EVENTS_PER_FRAME:]         # carried into the next frame, not discarded

        frame = events_to_frame_cropped(frame_events)
        frame_buffer.append(frame)

        accum = [leftover] if leftover.size > 0 else []
        accum_count = leftover.size

        if len(frame_buffer) == T:
            x_in = np.stack(frame_buffer)[:, None]           # [T, 1, 2, H, W]
            x_in = torch.from_numpy(x_in).float().to(DEVICE)

            with torch.no_grad():
                out_firing_rate = net(x_in).mean(0)
                pred = out_firing_rate.argmax(1).item()

            functional.reset_net(net)

            now = time.perf_counter()
            if last_pred_t is not None:
                latency_ms = (now - last_pred_t) * 1000.0
                prediction_latencies_ms.append(latency_ms)
                rows.append([len(prediction_latencies_ms), f'{latency_ms:.1f}', pred, CLASS_NAMES[pred]])
                print(f'prediction {len(prediction_latencies_ms)}/{N_PREDICTIONS}: '
                      f'{CLASS_NAMES[pred]}  (latency since last prediction: {latency_ms:.0f}ms)')
            else:
                print(f'first prediction: {CLASS_NAMES[pred]} (no latency yet -- nothing to measure against)')

            last_pred_t = now
            frame_buffer = []

            if len(prediction_latencies_ms) >= N_PREDICTIONS:
                break

    lat = np.array(prediction_latencies_ms)

    print()
    print(f'=== end-to-end pipeline latency (prediction-to-prediction), device={DEVICE} ===')
    print(f'predictions measured: {len(lat)}')
    print(f'latency: median={np.median(lat):.0f}ms  mean={lat.mean():.0f}ms  '
          f'std={lat.std():.0f}ms  min={lat.min():.0f}ms  max={lat.max():.0f}ms')
    print('(for reference: pure inference-only latency measured separately at ~356-384ms/window -- '
          'the gap between that and this number is event-accumulation time)')

    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['prediction_idx', 'latency_ms', 'pred_label', 'pred_class_name'])
        writer.writerows(rows)
    print(f'per-prediction log saved to {OUT_CSV}')


if __name__ == '__main__':
    main()
