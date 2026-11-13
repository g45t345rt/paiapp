"""Unit tests for asset organization and the Assets panel view model
(run: .venv\\Scripts\\python.exe tests\\test_assets_organize.py)."""

import os
import sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PIL import Image  # noqa: E402

from paintapp import assetview  # noqa: E402
from paintapp.document import AssetGroup, AssetLibrary, normalize_tags  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def make_asset(name, asset_id, source="imported", size=(10, 10), prompt="",
               tags=None, group=None, favorite=False):
    return SimpleNamespace(
        name=name,
        id=asset_id,
        source=source,
        prompt=prompt,
        tags=list(tags or []),
        group=group,
        favorite=favorite,
        image=Image.new("RGBA", size, (asset_id, 0, 0, 255)),
    )


def group(group_id, name, collapsed=False):
    return SimpleNamespace(id=group_id, name=name, collapsed=collapsed)


def ids(assets):
    return [asset.id for asset in assets]


# --------------------------------------------------------------- model
def test_asset_library_groups_and_moves():
    library = AssetLibrary()
    a = library.add_asset(name="a", image=Image.new("RGBA", (2, 2)))
    b = library.add_asset(name="b", image=Image.new("RGBA", (2, 2)))
    c = library.add_asset(name="c", image=Image.new("RGBA", (2, 2)))
    one = library.new_group("One")
    two = library.new_group("Two")
    check("groups created", [g.id for g in library.groups], [one.id, two.id])
    check("newest first", ids(library.assets), [c.id, b.id, a.id])

    library.move_asset(a, one.id)
    check("moved to group", a.group, one.id)
    check("moved to end of library", ids(library.assets), [c.id, b.id, a.id])

    library.move_asset(c, one.id, before=a)
    check("reordered before a", ids(library.assets), [b.id, c.id, a.id])
    check("both in group", [c.group, a.group], [one.id, one.id])

    library.move_asset(a, two.id)
    check("moved across groups", a.group, two.id)
    check("appended to group two", ids(library.assets), [b.id, c.id, a.id])

    library.move_asset(b, 9999)
    check("unknown group becomes ungrouped", b.group, None)

    moved = library.remove_group(one)
    check("remove_group reassigns members", moved, 1)
    check("member now ungrouped", c.group, None)
    check("group removed", [g.id for g in library.groups], [two.id])

    library.move_group(two, before=None)
    check("move_group keeps it", [g.id for g in library.groups], [two.id])


def test_asset_library_copy_and_clear():
    library = AssetLibrary()
    g = library.new_group("G")
    asset = library.add_asset(
        name="a", image=Image.new("RGBA", (2, 2)), group=g.id,
        tags=["  Red ", "red", "", "Blue"], favorite=True,
    )
    check("tags normalized", asset.tags, ["Blue", "Red"])
    clone = asset.copy()
    clone.tags.append("New")
    check("copy is independent", asset.tags, ["Blue", "Red"])
    check("copy keeps group", clone.group, g.id)
    check("copy keeps favorite", clone.favorite, True)
    check("find by id", library.find_by_id(asset.id) is asset, True)
    library.clear()
    check("clear removes groups", library.groups, [])
    check("clear resets group counter", library.group_counter, 0)


def test_normalize_tags():
    check("dedupes case-insensitively",
          normalize_tags(["b", "A", "a", " b "]), ["A", "b"])


# --------------------------------------------------------------- view model
def build_view():
    assets = [
        make_asset("Portrait sketch", 1, source="generated",
                   prompt="a portrait", tags=["sketch"], group=1),
        make_asset("Logo", 2, source="imported", size=(100, 50), group=1,
                   favorite=True),
        make_asset("Reference photo", 3, source="layer", group=None,
                   tags=["photo"]),
        make_asset("Old thing", 4, source="", size=(4, 4), group=2),
    ]
    groups = [group(1, "Sketches"), group(2, "Archive", collapsed=True)]
    return assets, groups


def test_filter_assets():
    assets, groups = build_view()
    state = assetview.AssetViewState()

    check("query by name", ids(assetview.filter_assets(assets, groups, state)),
          [1, 2, 3, 4])
    state.query = "portrait"
    check("query by tag/prompt/name",
          ids(assetview.filter_assets(assets, groups, state)), [1])
    state.query = "sketches"
    check("query by group name",
          ids(assetview.filter_assets(assets, groups, state)), [1, 2])
    state.query = ""

    state.sources = {"imported"}
    check("legacy source behaves as imported",
          ids(assetview.filter_assets(assets, groups, state)), [2, 4])
    state.sources = set()

    state.group_ids = {None}
    check("group filter for ungrouped",
          ids(assetview.filter_assets(assets, groups, state)), [3])
    state.group_ids = {1}
    check("group filter for a group",
          ids(assetview.filter_assets(assets, groups, state)), [1, 2])
    state.group_ids = set()

    state.tag = "PHOTO"
    check("tag filter is case-insensitive",
          ids(assetview.filter_assets(assets, groups, state)), [3])
    state.tag = None

    state.favorites_only = True
    check("favorites filter",
          ids(assetview.filter_assets(assets, groups, state)), [2])
    check("has_filters", state.has_filters(), True)
    state.clear_filters()
    check("clear_filters keeps sort",
          (state.has_filters(), state.sort), (False, assetview.SORT_MANUAL))


def test_sort_assets():
    assets, _groups = build_view()
    check("manual keeps order",
          ids(assetview.sort_assets(assets, assetview.SORT_MANUAL)), [1, 2, 3, 4])
    check("name",
          ids(assetview.sort_assets(assets, assetview.SORT_NAME)), [2, 4, 1, 3])
    check("name desc",
          ids(assetview.sort_assets(assets, assetview.SORT_NAME_DESC)), [3, 1, 4, 2])
    check("newest", ids(assetview.sort_assets(assets, assetview.SORT_NEWEST)),
          [4, 3, 2, 1])
    check("oldest", ids(assetview.sort_assets(assets, assetview.SORT_OLDEST)),
          [1, 2, 3, 4])
    check("largest", ids(assetview.sort_assets(assets, assetview.SORT_LARGEST)),
          [2, 1, 3, 4])
    check("smallest", ids(assetview.sort_assets(assets, assetview.SORT_SMALLEST)),
          [4, 1, 3, 2])
    check("source", ids(assetview.sort_assets(assets, assetview.SORT_SOURCE)),
          [1, 3, 2, 4])


def test_build_rows_groups_and_collapse():
    assets, groups = build_view()
    state = assetview.AssetViewState()

    def label(row):
        if row["kind"] == "group":
            return row["group"].name if row["group"] else "Ungrouped"
        return row["asset"].name

    rows = assetview.build_rows(assets, groups, state)
    kinds = [(row["kind"], label(row)) for row in rows]
    check("group headers and rows", kinds, [
        ("group", "Sketches"), ("asset", "Portrait sketch"), ("asset", "Logo"),
        ("group", "Archive"),
        ("group", "Ungrouped"), ("asset", "Reference photo"),
    ])
    check("collapsed group has no asset rows",
          rows[3]["collapsed"], True)
    check("archive asset hidden while collapsed",
          any(row.get("asset") and row["asset"].id == 4 for row in rows), False)

    state = assetview.AssetViewState()
    state.set_collapsed(groups[0], True)
    rows = assetview.build_rows(assets, groups, state)
    check("session collapse hides rows",
          any(row.get("asset") and row["asset"].id == 1 for row in rows), False)

    flat = assetview.build_rows(assets, [], assetview.AssetViewState())
    check("no groups means a flat list", [row["kind"] for row in flat],
          ["asset", "asset", "asset", "asset"])


def test_build_rows_filtering_and_group_filter():
    assets, groups = build_view()
    state = assetview.AssetViewState()
    state.query = "logo"
    rows = assetview.build_rows(assets, groups, state)
    check("empty groups hidden while filtering", [
        (row["kind"], row.get("group").name if row.get("group") else
         row.get("asset").name)
        for row in rows
    ], [
        ("group", "Sketches"), ("asset", "Logo"),
    ])

    state = assetview.AssetViewState()
    state.group_ids = {2}
    rows = assetview.build_rows(assets, groups, state)
    check("selected empty group still shown", [
        (row["kind"], row.get("group").name if row.get("group") else
         row.get("asset").name)
        for row in rows
    ], [("group", "Archive")])

    state = assetview.AssetViewState()
    state.sort = assetview.SORT_NAME
    rows = assetview.build_rows(assets, groups, state)
    group_rows = [row for row in rows if row.get("kind") == "group"]
    check("groups keep library order when sorting", [
        row["group"].name if row["group"] else "Ungrouped"
        for row in group_rows
    ], ["Sketches", "Archive", "Ungrouped"])


def main():
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
    if FAILED:
        print("\nFAILED:")
        for failure in FAILED:
            print("  " + failure)
        raise SystemExit(1)
    print("\nall asset organization tests passed")


if __name__ == "__main__":
    main()
