"""
HTTP-level and filesystem-safety tests for the browser dashboard
(scripts/live_dashboard.py), plus the powermetrics collector and the desktop
launcher helpers. The server runs in-process on an ephemeral 127.0.0.1 port
against a temporary REPO_ROOT, so nothing in the real repository is touched.
"""

from __future__ import annotations

import http.client
import io
import json
import os
import re
import sys
import tarfile
import threading
import zipfile
from pathlib import Path

import pytest


def _write_bids(root: Path, name: str = "x", subjects=("01",)) -> Path:
    for sub in subjects:
        (root / f"sub-{sub}" / "func").mkdir(parents=True, exist_ok=True)
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": name}), encoding="utf-8"
    )
    return root


def _tar_gz_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _bids_archive(name: str = "Uploaded") -> bytes:
    return _tar_gz_bytes(
        {
            "dataset_description.json": json.dumps({"Name": name}).encode(),
            "sub-01/func/sub-01_task-rest_bold.json": b"{}",
        }
    )


@pytest.fixture()
def dash_module(tmp_path, monkeypatch):
    import scripts.live_dashboard as dash

    monkeypatch.setattr(dash, "REPO_ROOT", tmp_path)
    monkeypatch.delenv("IMPACT_SYNTH_ROOT", raising=False)
    return dash


@pytest.fixture()
def state(dash_module, tmp_path):
    cfg = dash_module.DashboardConfig(
        out_dir=tmp_path / "outputs" / "scratch",
        dataset_id="ds003171",
        data_origin="real",
        subject_filter=None,
        refresh_sec=2.0,
        history_points=120,
    )
    return dash_module.DashboardState(cfg)


@pytest.fixture()
def server(dash_module, state):
    httpd = dash_module.make_server(state, "127.0.0.1", 0, csrf_token="test-token-123")
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd
    finally:
        httpd.shutdown()
        httpd.server_close()


def _request(httpd, method, path, *, body=None, headers=None, host=None):
    port = httpd.server_address[1]
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    hdrs = {"Host": host or f"127.0.0.1:{port}"}
    hdrs.update(headers or {})
    conn.request(method, path, body=body, headers=hdrs)
    resp = conn.getresponse()
    data = resp.read()
    headers_out = {k.lower(): v for k, v in resp.getheaders()}
    conn.close()
    return resp.status, data, headers_out


def _json_post(httpd, path, payload, *, token="test-token-123", extra=None, host=None):
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["X-IMPaCT-CSRF"] = token
    headers.update(extra or {})
    return _request(
        httpd, "POST", path, body=json.dumps(payload), headers=headers, host=host
    )


# -- Host / Origin / CSRF ------------------------------------------------------


def test_foreign_host_header_is_rejected_for_reads_and_writes(server):
    status, _, _ = _request(
        server, "GET", "/api/control/state", host="attacker.example"
    )
    assert status == 403
    status, _, _ = _request(
        server,
        "GET",
        "/api/control/state",
        host=f"attacker.example:{server.server_address[1]}",
    )
    assert status == 403
    status, _, _ = _request(server, "GET", "/api/control/state", host="127.0.0.1:1")
    assert status == 403  # wrong port is not our origin either
    status, _, _ = _json_post(
        server, "/api/errors/fix", {"action": "no_auto_fix"}, host="attacker.example"
    )
    assert status == 403
    status, body, _ = _request(server, "GET", "/api/control/state")
    assert status == 200
    assert "dataset_library" in json.loads(body)
    status, _, _ = _request(
        server,
        "GET",
        "/api/control/state",
        host=f"localhost:{server.server_address[1]}",
    )
    assert status == 200


def test_state_changing_post_requires_csrf_token(server):
    status, body, _ = _json_post(
        server, "/api/errors/fix", {"action": "no_auto_fix"}, token=None
    )
    assert status == 403
    assert "CSRF" in json.loads(body)["error"]
    status, _, _ = _json_post(
        server, "/api/errors/fix", {"action": "no_auto_fix"}, token="wrong"
    )
    assert status == 403
    status, body, _ = _json_post(server, "/api/errors/fix", {"action": "no_auto_fix"})
    assert status == 200
    assert json.loads(body)["ok"] is True


def test_cross_origin_and_non_json_posts_are_rejected(server):
    port = server.server_address[1]
    status, _, _ = _json_post(
        server,
        "/api/errors/fix",
        {"action": "no_auto_fix"},
        extra={"Origin": "https://evil.example"},
    )
    assert status == 403
    status, _, _ = _json_post(
        server,
        "/api/errors/fix",
        {"action": "no_auto_fix"},
        extra={"Origin": f"http://127.0.0.1:{port + 1}"},
    )
    assert status == 403
    status, _, _ = _json_post(
        server,
        "/api/errors/fix",
        {"action": "no_auto_fix"},
        extra={"Sec-Fetch-Site": "cross-site"},
    )
    assert status == 403
    status, _, _ = _json_post(
        server,
        "/api/errors/fix",
        {"action": "no_auto_fix"},
        extra={"Origin": f"http://127.0.0.1:{port}"},
    )
    assert status == 200
    # The original finding: a text/plain "simple" POST was parsed and executed.
    status, _, _ = _request(
        server,
        "POST",
        "/api/run/plan",
        body=json.dumps({"dataset_id": "ds003171"}),
        headers={"Content-Type": "text/plain", "X-IMPaCT-CSRF": "test-token-123"},
    )
    assert status == 415


def test_page_embeds_token_and_nonce_with_security_headers(server):
    status, body, headers = _request(server, "GET", "/")
    assert status == 200
    page = body.decode("utf-8")
    assert "__IMPACT_CSRF_TOKEN__" not in page and "__IMPACT_CSP_NONCE__" not in page
    assert 'content="test-token-123"' in page
    csp = headers["content-security-policy"]
    nonce = re.search(r"'nonce-([^']+)'", csp).group(1)
    assert f'<script nonce="{nonce}">' in page
    assert "frame-ancestors 'none'" in csp
    assert "unsafe-inline" not in csp.split("script-src", 1)[1].split(";", 1)[0]
    assert headers["x-frame-options"] == "DENY"
    assert headers["x-content-type-options"] == "nosniff"
    # A second load gets a fresh nonce.
    _, _, headers2 = _request(server, "GET", "/")
    assert headers2["content-security-policy"] != csp


def test_non_loopback_bind_requires_explicit_opt_in(dash_module, state, monkeypatch):
    with pytest.raises(ValueError, match="non-loopback"):
        dash_module.make_server(state, "0.0.0.0", 0)
    monkeypatch.setattr(
        sys, "argv", ["live_dashboard.py", "--host", "0.0.0.0", "--port", "0"]
    )
    with pytest.raises(SystemExit):
        dash_module.main()
    assert dash_module.is_loopback_host("127.0.0.1")
    assert dash_module.is_loopback_host("localhost")
    assert dash_module.is_loopback_host("[::1]")
    assert not dash_module.is_loopback_host("0.0.0.0")
    assert not dash_module.is_loopback_host("192.168.1.20")


# -- Dataset IDs, uploads and path containment ------------------------------------


@pytest.mark.parametrize(
    "bad_id", [".", "..", "...", "-rf", "a/b", "..%2f", "a b", "../data", ""]
)
def test_upload_with_unsafe_dataset_id_never_touches_data(server, tmp_path, bad_id):
    precious = tmp_path / "data" / "PRECIOUS.txt"
    precious.parent.mkdir(parents=True, exist_ok=True)
    precious.write_text("raw data", encoding="utf-8")
    synth_precious = tmp_path / "test_objects" / "KEEP.txt"
    synth_precious.parent.mkdir(parents=True, exist_ok=True)
    synth_precious.write_text("synthetic", encoding="utf-8")
    for origin in ("real", "dummy"):
        status, body, _ = _request(
            server,
            "POST",
            "/api/dataset/upload",
            body=_bids_archive(),
            headers={
                "X-IMPaCT-CSRF": "test-token-123",
                "X-Filename": "ds.tar.gz",
                "X-Dataset-Id": bad_id,
                "X-Data-Origin": origin,
                "X-Replace-Existing": "1",
                "Content-Type": "application/octet-stream",
            },
        )
        assert status == 400, body
    assert precious.read_text(encoding="utf-8") == "raw data"
    assert synth_precious.read_text(encoding="utf-8") == "synthetic"


def test_dataset_id_validation_rejects_traversal_and_flags(state):
    for bad in (".", "..", "...", "-x", "--hunter-stage", "a/b", "a\\b", "x" * 200):
        with pytest.raises(ValueError):
            state._clean_dataset_id(bad)
    assert state._clean_dataset_id("ds003171") == "ds003171"
    assert state._clean_dataset_id("ds005620_annex") == "ds005620_annex"


def test_upload_size_limit_is_enforced_before_reading(dash_module, state):
    httpd = dash_module.make_server(
        state, "127.0.0.1", 0, csrf_token="t", max_upload_bytes=1024
    )
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        status, body, _ = _request(
            httpd,
            "POST",
            "/api/dataset/upload",
            body=b"x" * 4096,
            headers={
                "X-IMPaCT-CSRF": "t",
                "X-Filename": "a.zip",
                "X-Dataset-Id": "dsbig",
            },
        )
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert status == 413
    assert "limit" in json.loads(body)["error"]
    staging = state._repo_root / "data" / "managed" / ".impact_upload_tmp"
    assert not staging.exists() or not any(staging.iterdir())


def test_reupload_requires_confirmation_and_replaces_atomically(server, tmp_path):
    def upload(name, replace):
        return _request(
            server,
            "POST",
            "/api/dataset/upload",
            body=_bids_archive(name),
            headers={
                "X-IMPaCT-CSRF": "test-token-123",
                "X-Filename": "dsup.tar.gz",
                "X-Dataset-Id": "dsup",
                "X-Data-Origin": "real",
                "X-Replace-Existing": "1" if replace else "0",
            },
        )

    status, body, _ = upload("First", replace=False)
    assert status == 200, body
    dest = tmp_path / "data" / "managed" / "dsup"
    assert (
        json.loads((dest / "dataset_description.json").read_text())["Name"] == "First"
    )
    status, body, _ = upload("Second", replace=False)
    assert status == 409
    assert "already exists" in json.loads(body)["error"]
    assert (
        json.loads((dest / "dataset_description.json").read_text())["Name"] == "First"
    )
    status, body, _ = upload("Second", replace=True)
    assert status == 200, body
    assert (
        json.loads((dest / "dataset_description.json").read_text())["Name"] == "Second"
    )
    staging = tmp_path / "data" / "managed" / ".impact_upload_tmp"
    assert not any(staging.iterdir())


def test_upload_refuses_symlinked_destination(state, tmp_path):
    target = tmp_path / "elsewhere" / "raw"
    target.mkdir(parents=True)
    (target / "KEEP.txt").write_text("keep", encoding="utf-8")
    managed = tmp_path / "data" / "managed"
    managed.mkdir(parents=True)
    (managed / "dslink").symlink_to(target, target_is_directory=True)
    archive = tmp_path / "a.tar.gz"
    archive.write_bytes(_bids_archive())
    with pytest.raises(ValueError, match="symbolic link"):
        state.upload_dataset_archive(
            archive_path=archive,
            dataset_id="dslink",
            out_dir=None,
            data_origin="real",
            modality_profile="auto",
            replace_existing=True,
        )
    assert (target / "KEEP.txt").read_text(encoding="utf-8") == "keep"


def test_archive_extraction_rejects_unsafe_members(state, tmp_path):
    target = tmp_path / "extract"
    target.mkdir()
    escape = tmp_path / "escape.tar.gz"
    escape.write_bytes(_tar_gz_bytes({"../evil.txt": b"x"}))
    with pytest.raises(ValueError, match="Unsafe archive member"):
        state._extract_archive_safe(escape, target)

    fifo = tmp_path / "fifo.tar"
    with tarfile.open(fifo, "w") as tf:
        info = tarfile.TarInfo("sub-01/pipe")
        info.type = tarfile.FIFOTYPE
        tf.addfile(info)
    with pytest.raises(ValueError, match="regular files"):
        state._extract_archive_safe(fifo, target)

    link = tmp_path / "link.tar"
    with tarfile.open(link, "w") as tf:
        info = tarfile.TarInfo("sub-01/link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tf.addfile(info)
    with pytest.raises(ValueError, match="Symlink"):
        state._extract_archive_safe(link, target)

    bomb = tmp_path / "bomb.zip"
    with zipfile.ZipFile(bomb, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("dataset_description.json", b"0" * 200_000)
    with pytest.raises(ValueError, match="extraction limit"):
        state._extract_archive_safe(bomb, target, max_total_bytes=10_000)
    assert not any(target.iterdir())


# -- Flag injection into run_pipeline argv -------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"sessions": ["awake", "--iim-checkpoint-dir", "/any/dir"]},
        {"subjects": ["01", "--hunter-campaign-dir", "/x"]},
        {"subjects": "-rf"},
        {"condition": "../../etc"},
        {"atlas": "*"},
        {"condition": "--no-ci"},
    ],
)
def test_run_command_rejects_flag_injection(state, tmp_path, overrides):
    root = _write_bids(tmp_path / "data" / "scratch" / "ds003171")
    payload = {
        "dataset_id": "ds003171",
        "bids_root": str(root),
        "run_preprocessing": True,
    }
    payload.update(overrides)
    with pytest.raises(ValueError, match="Invalid"):
        state._build_run_command(payload)


def test_run_command_argv_has_no_injected_options(state, tmp_path):
    root = _write_bids(tmp_path / "data" / "scratch" / "ds003171")
    cmd, _ = state._build_run_command(
        {
            "dataset_id": "ds003171",
            "bids_root": str(root),
            "sessions": "awake, deep",
            "subjects": ["sub-02CB", "10JR"],
            "run_preprocessing": True,
        }
    )
    options = [tok for tok in cmd[2:] if tok.startswith("-")]
    allowed = {
        "--dataset-id",
        "--data-origin",
        "--execution-mode",
        "--hardware-target",
        "--out-dir",
        "--bids-root",
        "--atlas",
        "--condition",
        "--sessions",
        "--subjects",
        "--mpc-metrics",
        "--run-preprocessing",
    }
    assert set(options) <= allowed
    i = cmd.index("--subjects")
    assert cmd[i + 1 : i + 3] == ["02CB", "10JR"]


# -- Stored XSS ----------


def test_library_and_checkpoint_templates_escape_untrusted_fields(
    dash_module, state, tmp_path
):
    root = tmp_path / "data" / "managed" / "dsxss"
    _write_bids(root, name='<img src=x onerror="alert(1)">')
    lib = {r["library_key"]: r for r in state.dataset_library()}
    # The server returns the raw value (JSON); escaping is the renderer's job.
    assert lib["dsxss"]["dataset_name"].startswith("<img")
    page = dash_module.HTML_PAGE
    for raw in (
        "${txt(rec.dataset_name",
        "${txt(rec.bids_root",
        "${c.file}",
        ">${rec.dataset_id}</div>",
        "${s.name}",
        "${txt(c.name",
        "${detail}</td>",
        'value="${d.dataset_id}"',
    ):
        assert raw not in page, raw
    for escaped in (
        "${esc(txt(rec.dataset_name",
        "${esc(txt(rec.bids_root",
        "${esc(c.file)}",
    ):
        assert escaped in page, escaped


def test_embedded_javascript_parses(dash_module, tmp_path):
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    js = dash_module.HTML_PAGE.split('<script nonce="__IMPACT_CSP_NONCE__">', 1)[
        1
    ].split("</script>", 1)[0]
    path = tmp_path / "page.js"
    path.write_text(js, encoding="utf-8")
    subprocess.run([node, "--check", str(path)], check=True, timeout=60)


# -- One-click fixes never pip-install silently ----------


def test_autofix_does_not_pip_install_without_explicit_confirmation(state, monkeypatch):
    import scripts.live_dashboard as dash

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))

        class R:
            returncode = 0
            stdout = ""
            stderr = ""

        return R()

    monkeypatch.setattr(dash.subprocess, "run", fake_run)
    real_preflight = state.setup_preflight

    def fake_preflight(payload):
        res = real_preflight(payload)
        res["report"]["missing_python_packages"] = ["definitely-not-installed-pkg"]
        return res

    monkeypatch.setattr(state, "setup_preflight", fake_preflight)
    res = state.setup_autofix({})
    assert not any("pip" in c for c in calls)
    assert res["pip_install_skipped"] == ["definitely-not-installed-pkg"]
    state._last_error = state._plain_error("ImportError: No module named foo")
    assert state._last_error["fix_action"] == "run_setup_check"
    state.apply_last_error_fix({"action": "run_setup_autofix"})
    assert not any("pip" in c for c in calls)
    assert (
        state._plain_error("something unrelated broke")["fix_action"]
        == "run_setup_check"
    )
    state.setup_autofix({"install_python_packages": True})
    assert any("pip" in c for c in calls)


# -- powermetrics collector ----------


def test_powermetrics_cache_is_chowned_back_to_sudo_user(tmp_path, monkeypatch):
    import scripts.powermetrics_telemetry as pm

    chowned = []
    monkeypatch.setattr(pm.os, "geteuid", lambda: 0, raising=False)
    monkeypatch.setattr(pm.os, "chown", lambda p, u, g: chowned.append((Path(p), u, g)))
    monkeypatch.setenv("SUDO_UID", "501")
    monkeypatch.setenv("SUDO_GID", "20")
    cache = tmp_path / "outputs" / "scratch" / "cache" / "powermetrics_telemetry.json"
    (tmp_path / "outputs").mkdir()
    pm._write_cache(cache, {"ok": True})
    paths = {p for p, _, _ in chowned}
    assert tmp_path / "outputs" / "scratch" in paths
    assert tmp_path / "outputs" / "scratch" / "cache" in paths
    assert tmp_path / "outputs" not in paths  # pre-existing folders are left alone
    assert any(p.name.endswith(".tmp") for p in paths)  # the file itself
    assert all((u, g) == (501, 20) for _, u, g in chowned)
    assert json.loads(cache.read_text())["ok"] is True

    chowned.clear()
    monkeypatch.setattr(pm.os, "geteuid", lambda: 501, raising=False)
    pm._write_cache(tmp_path / "other" / "c.json", {"ok": True})
    assert chowned == []


def test_powermetrics_parser_and_stdin_mode(tmp_path):
    import scripts.powermetrics_telemetry as pm

    sample = (
        "*** Sampled system activity (Mon) (5000.00ms elapsed) ***\n"
        "ANE die temperature: 41.0 C\n"
        "GPU die temperature: 55.0 C\n"
        "CPU Power: 3200 mW\n"
        "Current pressure level: Nominal\n"
    )
    parsed = pm._parse_powermetrics_output(sample)
    assert parsed["cpu_temp_c"] is None  # ANE/GPU are not reported as CPU temperature
    assert parsed["ane_temp_c"] == 41.0
    assert parsed["cpu_power_w"] == pytest.approx(3.2)
    stream = [
        line + "\n" for line in (sample + sample.replace("3200", "4100")).splitlines()
    ]
    cache = tmp_path / "cache" / "pm.json"
    n = pm._run_stdin_mode(cache, stream)
    assert n == 2
    assert json.loads(cache.read_text())["cpu_power_w"] == pytest.approx(4.1)


# -- desktop launcher ----------


def test_desktop_launcher_helpers_and_thread_safety(monkeypatch):
    pytest.importorskip("tkinter")
    import queue

    import scripts.impact_desktop_app as app

    assert app.is_loopback_host("127.0.0.1") and app.is_loopback_host("localhost")
    assert not app.is_loopback_host("0.0.0.0")
    kwargs = app.dashboard_popen_kwargs()
    assert "preexec_fn" not in kwargs
    if os.name == "posix":
        assert kwargs["start_new_session"] is True

    class NoTk:
        def set(self, *_a):
            raise AssertionError("Tk variable touched from a worker thread")

    launcher = app.DesktopLauncher.__new__(app.DesktopLauncher)
    launcher.log_q = queue.Queue()
    launcher.event_q = queue.Queue()
    launcher.status = NoTk()

    class FakeProc:
        stdout = iter(["line one\n"])

        def wait(self):
            return 0

    proc = FakeProc()
    launcher.proc = proc
    worker = threading.Thread(target=launcher._stream_proc, args=(proc,))
    worker.start()
    worker.join(5)
    assert launcher.event_q.get_nowait() == ("exited", proc)
    assert launcher.proc is proc  # cleared later on the Tk thread, not by the worker
