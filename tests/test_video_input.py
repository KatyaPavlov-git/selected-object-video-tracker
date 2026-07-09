"""Video input (A1): local-path OR network-stream URL passthrough.

Network-free by design — exercises only the URL-scheme classification and the
local-path guard. It never opens a real stream (the one open_video call that
uses a URL patches cv2.VideoCapture), so the suite has no network dependency.

Run: python3 -m unittest tests.test_video_input -v
"""
import unittest
from unittest import mock

from ground_target_tracking import utils


class TestVideoInputURL(unittest.TestCase):
    def test_stream_urls_recognized(self):
        for url in (
            "http://example.com/clip.mp4",
            "https://example.com/clip.mp4",
            "rtsp://cam.local/stream",
            "rtmp://server/live",
            "HTTP://EXAMPLE.COM/CLIP.MP4",  # case-insensitive
        ):
            self.assertTrue(utils.is_stream_url(url), url)

    def test_local_paths_not_urls(self):
        for p in (
            "videos/v8.mp4",
            "/absolute/path/clip.mp4",
            "./relative.mov",
            "clip.webm",
            "C:/windows/style.mp4",
            "",
        ):
            self.assertFalse(utils.is_stream_url(p), p)

    def test_missing_local_file_still_raises(self):
        # Regression: the local isfile gate stays intact for non-URL inputs.
        with self.assertRaises(ValueError):
            utils.open_video("/no/such/file/definitely_missing_9c2f.mp4")

    def test_url_bypasses_local_gate_without_network(self):
        # A URL must skip the "file not found" guard and proceed to VideoCapture.
        # cv2.VideoCapture is mocked so no network/stream is actually opened.
        fake_cap = mock.MagicMock()
        fake_cap.isOpened.return_value = True
        fake_cap.get.return_value = 0
        with mock.patch(
            "ground_target_tracking.utils.cv2.VideoCapture", return_value=fake_cap
        ):
            cap, meta = utils.open_video("http://example.com/stream.mp4")
        self.assertIs(cap, fake_cap)
        self.assertEqual(meta.path, "http://example.com/stream.mp4")


if __name__ == "__main__":
    unittest.main()
