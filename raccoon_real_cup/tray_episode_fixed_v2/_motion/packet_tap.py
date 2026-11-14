"""Experimental roboid 1.7.3 receive-side observation tap; no motor writes."""
import math
import queue
import threading
import time

class PacketTap:
    def __init__(self, roboid, log_event, capacity=4096):
        self.device = roboid
        self.log_event = log_event
        self.original = roboid._decode_sensory_packet
        self.items = queue.Queue(maxsize=capacity)
        self.done = threading.Event()
        self.count = 0
        self.dropped = 0
        self.errors = 0
        self.worker = threading.Thread(target=self._writer, name='packet-log', daemon=True)
        self.worker.start()
        roboid._decode_sensory_packet = self._decode

    def _decode(self, packet):
        result = self.original(packet)
        if result and not self.done.is_set():
            try:
                q = [0.0] * 4
                if self.device._encoder_device.read(q) != 4:
                    raise ValueError('encoder dimension')
                q = [float(v) for v in q]
                if not all(math.isfinite(v) for v in q):
                    raise ValueError('invalid encoder')
                item = (time.monotonic_ns(), q)
                self.items.put_nowait(item)
                self.count += 1
            except queue.Full:
                self.dropped += 1
            except Exception:
                self.errors += 1
        return result

    def _writer(self):
        while not self.done.is_set() or not self.items.empty():
            try:
                t, q = self.items.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                self.log_event('joint_packet', read_end_ns=t, joint_deg=q,
                               source='sdk_packet_decode_host_time')
            except Exception:
                self.errors += 1
            finally:
                self.items.task_done()

    def close(self):
        self.device._decode_sensory_packet = self.original
        self.done.set()
        self.worker.join()
