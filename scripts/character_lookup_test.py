"""Offline regressions for Chinese character identities, reference caching and tools."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import httpx

import smoke_test  # noqa: F401 - install the same AstrBot stubs as the core smoke suite
from comfyflow.clients.animadex import AnimaDexClient, _extract_text
from comfyflow.resources.characters import CharacterTagLookup, format_character_result, name_entries
from comfyflow.clients.external import DanbooruClient
from comfyflow.tools import ComfyuiLookupTool


OFFICIAL = {
    "hatsune_miku": {"zh_hans": "初音未来", "zh_hant": "初音未来", "ja": "初音ミク"},
    "elaina_(majo_no_tabitabi)": {"zh_hant": "伊蕾娜"},
    "wrong_person": {"zh_hans": "错误译名"},
    "person_a": {"zh_hans": "小明"},
    "person_b": {"zh_hans": "小明"},
}


class FailingMcp(AnimaDexClient):
    async def search_characters(self, *args, **kwargs):
        raise AssertionError("fast lookup must not wait for the legacy MCP server")


def attach_client(service, handler):
    service._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def check_lookup(root):
    calls = []

    async def handler(request):
        calls.append(str(request.url))
        name = request.url.path.rsplit("/", 1)[-1]
        if name == "character_official.json":
            return httpx.Response(200, json=OFFICIAL)
        if name == "character_manual.json":
            return httpx.Response(200, json={"wrong_person": "已校订译名"})
        if name == "zh_rejected.json":
            return httpx.Response(200, json={"wrong_person": "错误译名"})
        assert request.url.path == "/api/characters/search"
        await asyncio.sleep(0.01)
        return httpx.Response(
            200,
            json={
                "results": [
                    {"slug": "hatsune_miku_(append)", "tags": ["wrong_variant"]},
                    {
                        "slug": "hatsune_miku",
                        "tags": ["aqua hair", "twintails"],
                        "copyright": "vocaloid",
                    },
                ]
            },
        )

    service = CharacterTagLookup(root)
    attach_client(service, handler)
    # Cold concurrent callers share all three metadata downloads and one reference.
    first, second = await asyncio.gather(
        service.lookup("初音未来"), service.lookup("初音未来")
    )
    assert (
        len(first["matches"]) == 1
    )  # same normalized aliases must not duplicate a role
    assert first["matches"][0]["tag"] == "hatsune_miku"
    assert first["matches"][0]["appearance_tags"] == ["aqua hair", "twintails"]
    assert first == second and len(calls) == 4, calls
    for _ in range(3):
        assert (await service.lookup("初音未来"))["matches"][0][
            "copyright"
        ] == "vocaloid"
    assert len(calls) == 4
    ambiguous = await service.lookup("小明", limit=1)
    assert ambiguous["total"] == 2 and len(ambiguous["matches"]) == 1
    assert "候选" in format_character_result(ambiguous) and len(calls) == 4
    assert service.exact.get("错误译名") is None
    assert service.exact["已校订译名"][0]["tag"] == "wrong_person"
    assert not (await service.lookup("hatsune"))["exact"]
    tool = ComfyuiLookupTool(character_lookup=service, animadex=FailingMcp())
    result = await tool.call(None, type="character", query="初音未来")
    assert "角色 tag: hatsune_miku" in result and "外观参考 tag" in result
    assert len(calls) == 4
    await service.close()

    # Both identity and appearance remain available after restart without networking.
    restored = CharacterTagLookup(root, enabled=False)
    attach_client(
        restored,
        lambda request: (_ for _ in ()).throw(
            AssertionError("offline cache should not request")
        ),
    )
    assert (await restored.lookup("初音未来"))["matches"][0]["appearance_tags"] == [
        "aqua hair",
        "twintails",
    ]
    await restored.close()

    # A configured correction wins over the dataset, and is not baked into disk data.
    corrected = CharacterTagLookup(
        root,
        aliases=[
            {
                "name": "初音未来",
                "tag": "custom_miku",
                "appearance_tags": "blue_hair, twintails",
            }
        ],
    )
    attach_client(
        corrected,
        lambda request: (_ for _ in ()).throw(
            AssertionError("manual identity should not request")
        ),
    )
    assert (await corrected.lookup("初音未来"))["matches"][0]["tag"] == "custom_miku"
    await corrected._persist()
    assert json.loads(corrected.cache_path.read_text(encoding="utf-8"))["entries"]
    await corrected.close()
    cleared = CharacterTagLookup(root, enabled=False)
    assert (await cleared.lookup("初音未来"))["matches"][0]["tag"] == "hatsune_miku"
    await cleared.close()


async def check_failures(root):
    entries = name_entries(OFFICIAL, {}, {})
    names_file = root / "custom.json"
    names_file.write_text(json.dumps(entries), encoding="utf-8")
    service = CharacterTagLookup(root / "local", names_file=str(names_file))
    requests = 0

    async def handler(request):
        nonlocal requests
        requests += 1
        # A partial match must never supply appearance for the requested identity.
        return httpx.Response(
            200, json={"results": [{"slug": "hatsune_miku_(append)", "tags": ["bad"]}]}
        )

    attach_client(service, handler)
    result = await service.lookup("初音未来")
    assert result["matches"][0].get("appearance_tags") is None
    assert "外观标签暂无" in format_character_result(result)
    await service.lookup("初音未来")
    assert requests == 1  # negative reference results are briefly cached
    names_file.write_text(
        json.dumps(
            [
                {
                    "tag": "local_role",
                    "names": ["本地角色"],
                    "appearance_tags": ["red_hair"],
                }
            ]
        ),
        encoding="utf-8",
    )
    assert (await service.lookup("本地角色"))["matches"][0]["appearance_tags"] == [
        "red_hair"
    ]
    assert requests == 1
    await service.close()


async def check_booru():
    calls = []

    def handler(request):
        calls.append(request.url.path)
        assert "posts" not in request.url.path
        if request.url.path == "/tags.json":
            if request.url.params.get("search[name_matches]") == "*伊蕾娜*":
                return httpx.Response(200, json=[])
            return httpx.Response(
                200, json=[{"name": "hatsune_miku", "category": 4, "id": 1}]
            )
        assert request.url.path == "/wiki_pages.json"
        assert request.url.params["search[tag][category]"] == "4"
        return httpx.Response(
            200,
            json=[{"title": "elaina_(majo_no_tabitabi)", "other_names": ["伊蕾娜"]}],
        )

    client = DanbooruClient()
    client._client = httpx.AsyncClient(
        base_url="https://danbooru.donmai.us", transport=httpx.MockTransport(handler)
    )
    assert (await client.search_character("hatsune_miku", 0))["posts"] == []
    assert (await client.character_candidates("伊蕾娜"))[0][
        "tag"
    ] == "elaina_(majo_no_tabitabi)"
    assert calls == ["/tags.json", "/tags.json", "/wiki_pages.json"]
    await client.close()
    assert (
        _extract_text(
            [
                {
                    "result": {
                        "isError": True,
                        "content": [{"type": "text", "text": "failure"}],
                    }
                }
            ]
        )
        == ""
    )


async def check_learned_names_and_colors():
    class NewCharacterSource:
        count = 0

        async def character_candidates(self, query, limit):
            self.count += 1
            return [
                {"tag": "new_role", "names": [query], "source": "Danbooru Wiki 别名"}
            ]

    with tempfile.TemporaryDirectory() as temporary:
        source = NewCharacterSource()
        service = CharacterTagLookup(
            Path(temporary), enabled=False, danbooru=source, reference_url=""
        )
        assert (await service.lookup("新角色"))["matches"][0]["tag"] == "new_role"
        await service.lookup("新角色")
        assert source.count == 1
        await service.close()
        source.count = 0

        async def no_match(query, limit):
            source.count += 1
            return []

        source.character_candidates = no_match
        missing = CharacterTagLookup(
            Path(temporary) / "missing",
            enabled=False,
            danbooru=source,
            reference_url="",
        )
        assert not (await missing.lookup("未知角色"))["matches"]
        assert not (await missing.lookup("未知角色"))["matches"]
        assert source.count == 1
        await missing.close()
    formatted = format_character_result(
        {
            "query": "角色",
            "exact": True,
            "total": 1,
            "matches": [
                {
                    "tag": "role",
                    "appearance_tags": [
                        "long hair",
                        "blue eyes",
                        "purple eyes",
                        "white hair",
                        "grey hair",
                    ],
                },
            ],
        }
    )
    assert "眼色候选: blue eyes / purple eyes" in formatted
    assert "发色候选: white hair / grey hair" in formatted
    assert "外观参考 tag（本地角色库）: long hair" in formatted
    mixed = format_character_result(
        {
            "query": "角色",
            "exact": True,
            "total": 1,
            "matches": [
                {
                    "tag": "role",
                    "appearance_tags": [
                        "blue eyes",
                        "red eyes",
                        "heterochromia",
                        "pink hair",
                        "black hair",
                        "multicolored hair",
                    ],
                },
            ],
        }
    )
    assert "眼色候选" not in mixed and "发色候选" not in mixed
    assert "blue eyes, red eyes, heterochromia" in mixed


async def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        await check_lookup(root)
        await check_failures(root)
        await check_booru()
        await check_learned_names_and_colors()
    print(
        "character lookup identity / ambiguity / reference / cache / override / no-post regressions OK"
    )


if __name__ == "__main__":
    asyncio.run(main())
