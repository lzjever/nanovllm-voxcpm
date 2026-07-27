"""Unit tests for nanovllm_voxcpm.models.voxcpm2.server route/validation logic.

All tests run without a real GPU model.  The ``AsyncVoxCPM2ServerPool`` is
constructed via ``object.__new__`` (bypassing ``__init__``) and its ``servers``
list is replaced with lightweight fakes – the same technique used in
``test_voxcpm2_serverpool_lora_api.py``.
"""

from __future__ import annotations

import asyncio
import io

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Shared fake server helpers
# ---------------------------------------------------------------------------


class _FakeServer:
    """Minimal fake for AsyncVoxCPM2Server used inside pool tests."""

    def __init__(self):
        self.registered: list[tuple[str, str]] = []
        self.unregistered: list[str] = []
        self.generate_calls: list[dict] = []
        self.encode_calls: list[tuple[bytes, str, str]] = []

    async def register_lora(self, name: str, path: str):
        self.registered.append((name, path))
        return {"name": name}

    async def unregister_lora(self, name: str):
        self.unregistered.append(name)
        return {"name": name}

    async def get_model_info(self):
        return {
            "sample_rate": 48000,
            "encoder_sample_rate": 16000,
            "output_sample_rate": 48000,
            "channels": 1,
            "feat_dim": 4,
            "patch_size": 1,
            "model_path": "/fake",
        }

    async def encode_latents(self, wav: bytes, wav_format: str, role: str = "prompt") -> bytes:
        self.encode_calls.append((wav, wav_format, role))
        return np.zeros((4,), dtype=np.float32).tobytes()

    async def generate(
        self,
        target_text,
        prompt_latents=None,
        prompt_text="",
        max_generate_length=2000,
        temperature=1.0,
        cfg_value=2.0,
        ref_audio_latents=None,
        lora_name=None,
        seed=42,
    ):
        self.generate_calls.append(
            {
                "target_text": target_text,
                "lora_name": lora_name,
                "ref_audio_latents": ref_audio_latents,
            }
        )
        yield "chunk"


class _FailRegisterServer(_FakeServer):
    async def register_lora(self, name: str, path: str):
        raise RuntimeError("register_failed")


class _FailUnregisterServer(_FakeServer):
    async def unregister_lora(self, name: str):
        raise RuntimeError("unregister_failed")


# ---------------------------------------------------------------------------
# Helper: build a bare AsyncVoxCPM2ServerPool
# ---------------------------------------------------------------------------


def _make_pool(servers):
    from nanovllm_voxcpm.models.voxcpm2.server import AsyncVoxCPM2ServerPool

    pool = object.__new__(AsyncVoxCPM2ServerPool)
    pool.servers = list(servers)
    pool.servers_load = np.zeros(len(servers), dtype=np.int32)
    pool._prompt_pool = {}
    pool._registered_loras = set()
    pool._draining_loras = set()
    return pool


# ---------------------------------------------------------------------------
# gen_uuid
# ---------------------------------------------------------------------------


def test_gen_uuid_returns_hex_string():
    from nanovllm_voxcpm.models.voxcpm2.server import gen_uuid

    uid = gen_uuid()
    assert isinstance(uid, str)
    assert len(uid) == 32
    assert uid != gen_uuid()


# ---------------------------------------------------------------------------
# VoxCPM2ServerImpl – unit-level (no engine)
# ---------------------------------------------------------------------------


class _FakeLLM:
    feat_dim = 4
    patch_size = 1

    def __init__(self):
        self.added_requests: list[dict] = []
        self.registered: list[tuple[str, str]] = []
        self.unregistered: list[str] = []
        self.loras: list = []

    def add_request(self, **kwargs):
        self.added_requests.append(kwargs)

    def register_lora(self, name: str, path: str):
        self.registered.append((name, path))

    def unregister_lora(self, name: str):
        self.unregistered.append(name)

    def list_loras(self):
        return self.loras

    def cancel_sequence(self, seq_id: str):
        pass

    def step(self):
        return []

    def is_finished(self):
        return True


def _make_server_impl(feat_dim: int = 4, patch_size: int = 1):
    from nanovllm_voxcpm.models.voxcpm2.server import VoxCPM2ServerImpl

    srv = VoxCPM2ServerImpl.__new__(VoxCPM2ServerImpl)
    srv.encoder_sample_rate = 16000
    srv.output_sample_rate = 48000
    srv.model_path = "/fake/model"
    llm = _FakeLLM()
    llm.feat_dim = feat_dim
    llm.patch_size = patch_size
    srv.llm = llm
    return srv


def test_server_impl_health():
    srv = _make_server_impl()
    assert srv.health() == {"status": "ok"}


def test_server_impl_get_model_info():
    srv = _make_server_impl(feat_dim=8, patch_size=2)
    info = srv.get_model_info()
    assert info["sample_rate"] == 48000
    assert info["encoder_sample_rate"] == 16000
    assert info["output_sample_rate"] == 48000
    assert info["feat_dim"] == 8
    assert info["patch_size"] == 2
    assert info["model_path"] == "/fake/model"
    assert info["channels"] == 1


def test_server_impl_register_and_list_and_unregister_lora():
    from nanovllm_voxcpm.models.voxcpm2.server import LoRAInfo

    srv = _make_server_impl()

    class _LLMEntry:
        def __init__(self, name):
            self.name = name

    srv.llm.loras = [_LLMEntry("beta")]
    srv.register_lora("beta", "/tmp/beta")
    assert srv.llm.registered == [("beta", "/tmp/beta")]

    loras = srv.list_loras()
    assert loras == [LoRAInfo(name="beta")]

    srv.unregister_lora("beta")
    assert "beta" in srv.llm.unregistered


def test_server_impl_cancel_and_step_and_is_finished():
    srv = _make_server_impl()
    srv.cancel("some-seq-id")
    assert srv.step() == []
    assert srv.is_finished() is True


# ---------------------------------------------------------------------------
# VoxCPM2ServerImpl.add_request – validation paths
# ---------------------------------------------------------------------------


def test_add_request_no_prompt_latents_no_prompt_text_ok():
    srv = _make_server_impl()
    srv.add_request("seq1", "hello")
    assert len(srv.llm.added_requests) == 1
    req = srv.llm.added_requests[0]
    assert req["seq_id"] == "seq1"
    assert req["target_text"] == "hello"
    assert req["prompt_text"] == ""


def test_add_request_no_prompt_latents_with_prompt_text_raises():
    srv = _make_server_impl()
    with pytest.raises(ValueError, match="Prompt text is not allowed"):
        srv.add_request("seq1", "hello", prompt_text="some text")


def test_add_request_with_prompt_latents_and_prompt_text_ok():
    srv = _make_server_impl(feat_dim=4)
    latents_bytes = np.ones((2, 4), dtype=np.float32).tobytes()
    srv.add_request("seq1", "hello", prompt_latents=latents_bytes, prompt_text="world")
    assert len(srv.llm.added_requests) == 1
    req = srv.llm.added_requests[0]
    assert req["prompt_text"] == "world"
    assert req["prompt_latents"].shape == (2, 4)


def test_add_request_with_prompt_latents_but_no_prompt_text_raises():
    srv = _make_server_impl(feat_dim=4)
    latents_bytes = np.ones((2, 4), dtype=np.float32).tobytes()
    with pytest.raises(ValueError, match="Prompt text is required"):
        srv.add_request("seq1", "hello", prompt_latents=latents_bytes)


def test_add_request_passes_ref_audio_latents_without_prompt():
    """ref_audio_latents may be passed even without prompt_latents."""
    srv = _make_server_impl(feat_dim=4)
    ref = np.ones((3, 4), dtype=np.float32).tobytes()
    srv.add_request("seq1", "hello", ref_audio_latents=ref)
    req = srv.llm.added_requests[0]
    assert req["ref_audio_latents"] is not None
    assert req["ref_audio_latents"].shape == (3, 4)


def test_add_request_passes_ref_audio_latents_with_prompt():
    """ref_audio_latents may also accompany prompt_latents + prompt_text."""
    srv = _make_server_impl(feat_dim=4)
    lat = np.ones((2, 4), dtype=np.float32).tobytes()
    ref = np.ones((3, 4), dtype=np.float32).tobytes()
    srv.add_request("seq1", "hello", prompt_latents=lat, prompt_text="p", ref_audio_latents=ref)
    req = srv.llm.added_requests[0]
    assert req["ref_audio_latents"] is not None
    assert req["ref_audio_latents"].shape == (3, 4)


def test_add_request_ref_audio_latents_none_is_forwarded():
    """When ref_audio_latents=None the engine receives None."""
    srv = _make_server_impl()
    srv.add_request("seq1", "hello", ref_audio_latents=None)
    req = srv.llm.added_requests[0]
    assert req.get("ref_audio_latents") is None


def test_add_request_passes_lora_name_and_seed():
    srv = _make_server_impl()
    srv.add_request("seq1", "hi", lora_name="my_lora", seed=99)
    req = srv.llm.added_requests[0]
    assert req["lora_name"] == "my_lora"
    assert req["seed"] == 99


# ---------------------------------------------------------------------------
# VoxCPM2ServerImpl.encode_latents – stereo path
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")


def test_encode_latents_converts_stereo_to_mono(monkeypatch):
    from nanovllm_voxcpm.models.voxcpm2.server import VoxCPM2ServerImpl
    import librosa

    srv = VoxCPM2ServerImpl.__new__(VoxCPM2ServerImpl)
    srv.encoder_sample_rate = 16000

    captured = {}

    class _LLM:
        patch_size = 1

        def encode_latents(self, wav_tensor, role):
            captured["shape"] = wav_tensor.shape
            captured["role"] = role
            return np.zeros((2, 4), dtype=np.float32)

    srv.llm = _LLM()

    # Return stereo (2, N) audio.
    def _fake_librosa_load(file_obj, sr, mono):
        return np.zeros((2, 160), dtype=np.float32), sr

    monkeypatch.setattr("nanovllm_voxcpm.models.voxcpm2.server.librosa.load", _fake_librosa_load)

    srv.encode_latents(b"fake-wav", "wav")

    # Stereo → mono: channel dim should become 1.
    assert captured["shape"][0] == 1
    assert captured["role"] == "prompt"


def test_encode_latents_mono_keeps_shape(monkeypatch):
    from nanovllm_voxcpm.models.voxcpm2.server import VoxCPM2ServerImpl

    srv = VoxCPM2ServerImpl.__new__(VoxCPM2ServerImpl)
    srv.encoder_sample_rate = 16000

    captured = {}

    class _LLM:
        patch_size = 1

        def encode_latents(self, wav_tensor, role):
            captured["ndim"] = wav_tensor.ndim
            captured["shape"] = wav_tensor.shape
            captured["role"] = role
            return np.zeros((2, 4), dtype=np.float32)

    srv.llm = _LLM()

    # Return 1-D array (librosa mono).
    def _fake_librosa_load(file_obj, sr, mono):
        return np.zeros((160,), dtype=np.float32), sr

    monkeypatch.setattr("nanovllm_voxcpm.models.voxcpm2.server.librosa.load", _fake_librosa_load)

    out = srv.encode_latents(b"fake-wav", "wav")
    assert isinstance(out, bytes)
    # After unsqueeze the tensor should be (1, N).
    assert captured["shape"][0] == 1
    assert captured["role"] == "prompt"


# ---------------------------------------------------------------------------
# AsyncVoxCPM2ServerPool – get_model_info empty-pool guard
# ---------------------------------------------------------------------------


async def _pool_get_model_info_empty():
    pool = _make_pool([])
    with pytest.raises(RuntimeError, match="empty"):
        await pool.get_model_info()


def test_pool_get_model_info_raises_on_empty_pool():
    asyncio.run(_pool_get_model_info_empty())


async def _pool_get_model_info_single_server():
    pool = _make_pool([_FakeServer()])
    info = await pool.get_model_info()
    assert info["output_sample_rate"] == 48000


def test_pool_get_model_info_delegates_to_first_server():
    asyncio.run(_pool_get_model_info_single_server())


# ---------------------------------------------------------------------------
# AsyncVoxCPM2ServerPool – register_lora guards
# ---------------------------------------------------------------------------


async def _pool_register_lora_duplicate():
    pool = _make_pool([_FakeServer()])
    pool._registered_loras = {"demo"}
    with pytest.raises(ValueError, match="already registered"):
        await pool.register_lora("demo", "/tmp/demo")


def test_pool_register_lora_duplicate_raises():
    asyncio.run(_pool_register_lora_duplicate())


async def _pool_register_lora_draining_guard():
    pool = _make_pool([_FakeServer()])
    pool._draining_loras = {"demo"}
    with pytest.raises(ValueError, match="already registered"):
        await pool.register_lora("demo", "/tmp/demo")


def test_pool_register_lora_draining_guard_raises():
    asyncio.run(_pool_register_lora_draining_guard())


async def _pool_register_lora_rolls_back_on_failure():
    good = _FakeServer()
    bad = _FailRegisterServer()
    pool = _make_pool([good, bad])

    with pytest.raises(RuntimeError, match="register_failed"):
        await pool.register_lora("alpha", "/tmp/alpha")

    assert "alpha" not in pool._registered_loras
    assert "alpha" in good.unregistered


def test_pool_register_lora_rolls_back_on_failure():
    asyncio.run(_pool_register_lora_rolls_back_on_failure())


# ---------------------------------------------------------------------------
# AsyncVoxCPM2ServerPool – unregister_lora guards
# ---------------------------------------------------------------------------


async def _pool_unregister_not_registered():
    pool = _make_pool([_FakeServer()])
    with pytest.raises(ValueError, match="not registered"):
        await pool.unregister_lora("missing")


def test_pool_unregister_lora_not_registered_raises():
    asyncio.run(_pool_unregister_not_registered())


async def _pool_unregister_already_draining():
    pool = _make_pool([_FakeServer()])
    pool._registered_loras = {"demo"}
    pool._draining_loras = {"demo"}
    with pytest.raises(ValueError, match="already draining"):
        await pool.unregister_lora("demo")


def test_pool_unregister_lora_already_draining_raises():
    asyncio.run(_pool_unregister_already_draining())


async def _pool_unregister_success():
    pool = _make_pool([_FakeServer(), _FakeServer()])
    pool._registered_loras = {"demo"}
    result = await pool.unregister_lora("demo")
    assert result == {"name": "demo"}
    assert "demo" not in pool._registered_loras
    assert "demo" not in pool._draining_loras


def test_pool_unregister_lora_success():
    asyncio.run(_pool_unregister_success())


# ---------------------------------------------------------------------------
# AsyncVoxCPM2ServerPool – encode_latents routes to min-load server
# ---------------------------------------------------------------------------


async def _pool_encode_latents_routes_to_min_load():
    s0 = _FakeServer()
    s1 = _FakeServer()
    pool = _make_pool([s0, s1])
    pool.servers_load[0] = 5  # s1 has lower load

    out = await pool.encode_latents(b"data", "wav", role="reference")
    assert isinstance(out, bytes)
    assert s1.encode_calls == [(b"data", "wav", "reference")]
    assert len(s0.encode_calls) == 0


def test_pool_encode_latents_routes_to_min_load_server():
    asyncio.run(_pool_encode_latents_routes_to_min_load())


# ---------------------------------------------------------------------------
# AsyncVoxCPM2ServerPool – add_prompt and remove_prompt
# ---------------------------------------------------------------------------


async def _pool_add_and_remove_prompt():
    pool = _make_pool([_FakeServer()])

    prompt_id = await pool.add_prompt(b"wav_bytes", "wav", "some text")
    assert isinstance(prompt_id, str)
    assert prompt_id in pool._prompt_pool
    assert pool._prompt_pool[prompt_id]["text"] == "some text"

    await pool.remove_prompt(prompt_id)
    assert prompt_id not in pool._prompt_pool


def test_pool_add_and_remove_prompt():
    asyncio.run(_pool_add_and_remove_prompt())


# ---------------------------------------------------------------------------
# AsyncVoxCPM2ServerPool – generate with prompt_id validation
# ---------------------------------------------------------------------------


async def _pool_generate_unknown_prompt_id():
    pool = _make_pool([_FakeServer()])
    with pytest.raises(ValueError, match="not found"):
        async for _ in pool.generate("hello", prompt_id="bad-id"):
            pass


def test_pool_generate_unknown_prompt_id_raises():
    asyncio.run(_pool_generate_unknown_prompt_id())


async def _pool_generate_prompt_id_and_latents_conflict():
    pool = _make_pool([_FakeServer()])
    pool._prompt_pool["pid1"] = {"latents": b"", "text": "t"}
    with pytest.raises(ValueError, match="cannot be provided at the same time"):
        async for _ in pool.generate(
            "hello",
            prompt_id="pid1",
            prompt_latents=np.zeros((1, 4), dtype=np.float32).tobytes(),
        ):
            pass


def test_pool_generate_prompt_id_and_latents_conflict_raises():
    asyncio.run(_pool_generate_prompt_id_and_latents_conflict())


async def _pool_generate_prompt_id_and_text_conflict():
    pool = _make_pool([_FakeServer()])
    pool._prompt_pool["pid1"] = {"latents": b"", "text": "t"}
    with pytest.raises(ValueError, match="cannot be provided at the same time"):
        async for _ in pool.generate("hello", prompt_id="pid1", prompt_text="extra"):
            pass


def test_pool_generate_prompt_id_and_text_conflict_raises():
    asyncio.run(_pool_generate_prompt_id_and_text_conflict())


async def _pool_generate_resolves_prompt_id():
    s = _FakeServer()
    pool = _make_pool([s])
    lat = np.zeros((2, 4), dtype=np.float32).tobytes()
    pool._prompt_pool["pid1"] = {"latents": lat, "text": "my-text"}

    chunks = []
    async for chunk in pool.generate("hello", prompt_id="pid1"):
        chunks.append(chunk)

    assert chunks == ["chunk"]
    assert s.generate_calls[0]["target_text"] == "hello"


def test_pool_generate_resolves_prompt_id():
    asyncio.run(_pool_generate_resolves_prompt_id())


async def _pool_generate_forwards_ref_audio_latents():
    s = _FakeServer()
    pool = _make_pool([s])
    ref = np.ones((2, 4), dtype=np.float32).tobytes()

    chunks = []
    async for chunk in pool.generate("hello", ref_audio_latents=ref):
        chunks.append(chunk)

    assert chunks == ["chunk"]
    assert s.generate_calls[0]["ref_audio_latents"] == ref


def test_pool_generate_forwards_ref_audio_latents():
    asyncio.run(_pool_generate_forwards_ref_audio_latents())


async def _pool_generate_lora_draining_rejected():
    pool = _make_pool([_FakeServer()])
    pool._registered_loras = {"lora1"}
    pool._draining_loras = {"lora1"}
    with pytest.raises(ValueError, match="not registered"):
        async for _ in pool.generate("hello", lora_name="lora1"):
            pass


def test_pool_generate_lora_draining_rejected():
    asyncio.run(_pool_generate_lora_draining_rejected())


async def _pool_generate_updates_load_counter():
    s = _FakeServer()
    pool = _make_pool([s])

    chunks = []
    async for chunk in pool.generate("hello"):
        assert pool.servers_load[0] == 1
        chunks.append(chunk)

    assert pool.servers_load[0] == 0
    assert chunks == ["chunk"]


def test_pool_generate_updates_load_counter():
    asyncio.run(_pool_generate_updates_load_counter())


# ---------------------------------------------------------------------------
# AsyncVoxCPM2ServerPool – wait_for_ready and stop
# ---------------------------------------------------------------------------


class _FakeServerWithReadyStop(_FakeServer):
    def __init__(self):
        super().__init__()
        self.ready_called = False
        self.stop_called = False

    async def wait_for_ready(self):
        self.ready_called = True

    async def stop(self):
        self.stop_called = True


async def _pool_wait_for_ready_and_stop():
    s0 = _FakeServerWithReadyStop()
    s1 = _FakeServerWithReadyStop()
    pool = _make_pool([s0, s1])

    await pool.wait_for_ready()
    assert s0.ready_called and s1.ready_called

    await pool.stop()
    assert s0.stop_called and s1.stop_called


def test_pool_wait_for_ready_and_stop():
    asyncio.run(_pool_wait_for_ready_and_stop())


def test_async_server_rejects_unknown_kwargs_before_starting_process():
    from nanovllm_voxcpm.models.voxcpm2.server import AsyncVoxCPM2Server

    with pytest.raises(ValueError, match="Unknown kwargs"):
        AsyncVoxCPM2Server("/fake/model", unsupported=True)


def test_async_server_pool_rejects_unknown_kwargs_before_starting_servers():
    from nanovllm_voxcpm.models.voxcpm2.server import AsyncVoxCPM2ServerPool

    with pytest.raises(ValueError, match="Unknown kwargs"):
        AsyncVoxCPM2ServerPool("/fake/model", unsupported=True)


async def _async_server_forwards_control_plane_calls():
    from nanovllm_voxcpm.models.voxcpm2.server import AsyncVoxCPM2Server

    server = object.__new__(AsyncVoxCPM2Server)
    calls = []

    async def submit(command, *args):
        calls.append((command, args))
        return command

    server.submit = submit

    assert await server.health() == "health"
    assert await server.get_model_info() == "get_model_info"
    assert await server.encode_latents(b"wav", "wav") == "encode_latents"
    assert await server.register_lora("demo", "/tmp/demo") == "register_lora"
    assert await server.unregister_lora("demo") == "unregister_lora"
    assert await server.list_loras() == "list_loras"
    assert [command for command, _ in calls] == [
        "health",
        "get_model_info",
        "encode_latents",
        "register_lora",
        "unregister_lora",
        "list_loras",
    ]


def test_async_server_forwards_control_plane_calls():
    asyncio.run(_async_server_forwards_control_plane_calls())


async def _async_server_generate_completes_and_cleans_stream_state():
    from nanovllm_voxcpm.models.voxcpm2.server import AsyncVoxCPM2Server, GenerationCompletion

    server = object.__new__(AsyncVoxCPM2Server)
    server.stream_table = {}
    commands = []

    async def submit(command, *args):
        commands.append(command)
        if command == "add_request":
            stream = server.stream_table[args[0]]
            await stream.put(np.array([1.0], dtype=np.float32))
            await stream.put(GenerationCompletion(type="completion", generated_latents=b"latents"))
            await stream.put(None)

    server.submit = submit

    chunks = [chunk async for chunk in server.generate("hello")]

    assert len(chunks) == 2
    assert chunks[0].tolist() == [1.0]
    assert chunks[1] == {"type": "completion", "generated_latents": b"latents"}
    assert commands == ["add_request"]
    assert server.stream_table == {}


def test_async_server_generate_completes_and_cleans_stream_state():
    asyncio.run(_async_server_generate_completes_and_cleans_stream_state())


async def _async_server_generate_cancels_when_consumer_closes_early():
    from nanovllm_voxcpm.models.voxcpm2.server import AsyncVoxCPM2Server

    server = object.__new__(AsyncVoxCPM2Server)
    server.stream_table = {}
    commands = []

    async def submit(command, *args):
        commands.append(command)
        if command == "add_request":
            await server.stream_table[args[0]].put(np.array([1.0], dtype=np.float32))

    server.submit = submit
    stream = server.generate("hello")

    await stream.__anext__()
    await stream.aclose()

    assert commands == ["add_request", "cancel"]
    assert server.stream_table == {}


def test_async_server_generate_cancels_when_consumer_closes_early():
    asyncio.run(_async_server_generate_cancels_when_consumer_closes_early())


async def _pool_generate_closes_inner_stream_on_outer_close():
    class _CloseAwareServer(_FakeServer):
        def __init__(self):
            super().__init__()
            self.closed = False

        async def generate(self, *args, **kwargs):
            try:
                yield np.array([1.0], dtype=np.float32)
                await asyncio.Event().wait()
            finally:
                self.closed = True

    server = _CloseAwareServer()
    pool = _make_pool([server])
    stream = pool.generate("hello")

    await stream.__anext__()
    await stream.aclose()

    assert server.closed is True
    assert pool.servers_load.tolist() == [0]


def test_pool_generate_closes_inner_stream_on_outer_close():
    asyncio.run(_pool_generate_closes_inner_stream_on_outer_close())


async def _async_server_fatal_error_unblocks_active_stream():
    from nanovllm_voxcpm.models.voxcpm2.server import AsyncVoxCPM2Server

    server = object.__new__(AsyncVoxCPM2Server)
    server._queue_out_async = asyncio.Queue()
    loop = asyncio.get_running_loop()
    server._init_fut = loop.create_future()
    server._init_fut.set_result(None)
    pending = loop.create_future()
    server.op_table = {"pending": pending}
    server.stream_table = {"seq": asyncio.Queue()}
    server._fatal_error = None
    recv_task = asyncio.create_task(server.recv_queue())

    await server._queue_out_async.put({"type": "fatal_error", "error": "engine failed"})

    failure = await asyncio.wait_for(server.stream_table["seq"].get(), timeout=0.1)
    await recv_task

    assert isinstance(failure, RuntimeError)
    assert "engine failed" in str(failure)
    with pytest.raises(RuntimeError, match="engine failed"):
        await pending
    with pytest.raises(RuntimeError, match="engine failed"):
        await server.submit("health")


def test_async_server_fatal_error_unblocks_active_stream():
    asyncio.run(_async_server_fatal_error_unblocks_active_stream())


async def _async_server_generate_raises_fatal_without_cancel():
    from nanovllm_voxcpm.models.voxcpm2.server import AsyncVoxCPM2Server

    server = object.__new__(AsyncVoxCPM2Server)
    server.stream_table = {}
    server._fatal_error = None
    commands = []

    async def submit(command, *args):
        commands.append(command)
        if command == "add_request":
            server._fatal_error = "engine failed"
            await server.stream_table[args[0]].put(RuntimeError("engine failed"))

    server.submit = submit
    stream = server.generate("hello")

    with pytest.raises(RuntimeError, match="engine failed"):
        await stream.__anext__()

    assert commands == ["add_request"]
    assert server.stream_table == {}


def test_async_server_generate_raises_fatal_without_cancel():
    asyncio.run(_async_server_generate_raises_fatal_without_cancel())


def test_main_loop_reports_step_exception_as_fatal(monkeypatch):
    from queue import Queue

    from nanovllm_voxcpm.models.voxcpm2 import server as server_module

    class _FailingServer:
        def health(self):
            return {"status": "ok"}

        def is_finished(self):
            return False

        def step(self):
            raise RuntimeError("step failed")

    monkeypatch.setattr(server_module, "VoxCPM2ServerImpl", lambda *args, **kwargs: _FailingServer())
    monkeypatch.setattr(server_module.signal, "signal", lambda *args: None)
    queue_in = Queue()
    queue_out = Queue()
    queue_in.put({"id": "health", "type": "health", "args": (), "kwargs": {}})

    server_module.main_loop(queue_in, queue_out, (), {})

    messages = [queue_out.get_nowait() for _ in range(queue_out.qsize())]
    assert [message["type"] for message in messages] == ["init_ok", "response", "fatal_error"]
    assert "step failed" in messages[-1]["error"]


def test_queue_bridge_reports_unexpected_child_exit():
    from queue import Empty

    from nanovllm_voxcpm.models.voxcpm2.server import AsyncVoxCPM2Server

    class _EmptyQueue:
        def get(self, timeout):
            raise Empty

    class _Loop:
        def __init__(self):
            self.messages = []

        def call_soon_threadsafe(self, callback, message):
            callback(message)

    class _Stop:
        def __init__(self):
            self.calls = 0

        def is_set(self):
            self.calls += 1
            return self.calls > 1

    server = object.__new__(AsyncVoxCPM2Server)
    server.queue_out = _EmptyQueue()
    server.process = type("_Process", (), {"exitcode": 17})()
    server._queue_out_stop = _Stop()
    server._queue_out_async = asyncio.Queue()
    loop = _Loop()

    server._queue_out_bridge(loop)

    message = server._queue_out_async.get_nowait()
    assert message["type"] == "fatal_error"
    assert "exitcode=17" in message["error"]
