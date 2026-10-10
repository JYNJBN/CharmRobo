import unittest

from app.integrations.hardware_audio import (
    HardwareAudioError,
    build_ogg_speex,
    parse_ci_speex_frames,
)


class HardwareAudioTest(unittest.TestCase):
    def test_parse_ci_speex_frames(self) -> None:
        first = b"a" * 42
        second = b"b" * 42
        payload = bytes([len(first)]) + first + bytes([len(second)]) + second

        self.assertEqual(parse_ci_speex_frames(payload), [first, second])

    def test_rejects_truncated_frame(self) -> None:
        with self.assertRaises(HardwareAudioError):
            parse_ci_speex_frames(bytes([5]) + b"abc")

    def test_builds_ogg_speex(self) -> None:
        ogg = build_ogg_speex([b"a" * 42])

        self.assertTrue(ogg.startswith(b"OggS"))
        self.assertGreater(ogg.count(b"OggS"), 1)


if __name__ == "__main__":
    unittest.main()
