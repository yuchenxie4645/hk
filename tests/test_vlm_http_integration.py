import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np

from app.config import Settings
from app.facade import LocalRecallFacadeImpl
from scripts.mock_vlm_server import Handler


def test_manual_pair_uses_openai_http_contract_and_persists_evidence(tmp_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    facade = None
    try:
        settings = Settings(
            data_dir=str(tmp_path / "data"),
            database_path=str(tmp_path / "data" / "memory.db"),
            gradio_temp_dir=str(tmp_path / "data" / "tmp"),
            vlm_base_url=f"http://127.0.0.1:{server.server_port}/v1",
            inference_mode="real",
            offline_mode=True,
        )
        facade = LocalRecallFacadeImpl(settings, mock=False)
        facade.add_zone("Left Zone", "left_zone", 0, 0, 0.3, 1)
        facade.add_zone("Right Zone", "right_zone", 0.7, 0, 1, 1)
        facade.enroll_object("Red Cube", "cube", np.full((80, 80, 3), 180, dtype=np.uint8))
        event = facade.analyze_pair(
            np.zeros((80, 120, 3), dtype=np.uint8),
            np.full((80, 120, 3), 255, dtype=np.uint8),
        )
        assert event.accepted
        assert event.model_name == settings.vlm_model
        assert event.object_id == "red_cube"
        assert event.from_zone_id == "left_zone"
        assert event.to_zone_id == "right_zone"
        assert all(
            path and Path(path).exists()
            for path in (event.pre_image_path, event.post_image_path, event.composite_path)
        )
        assert facade.get_status().model_ready
    finally:
        if facade is not None:
            facade.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

