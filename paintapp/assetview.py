"""Pure filtering, sorting and grouping for the Assets panel.

The panel is a Tk widget, which makes its view logic awkward to test. This
module holds the parts that decide *what* is shown - the query, the filters, the
sort order and the group headers - so they can be exercised without a display.
It knows nothing about Tkinter; :class:`paintapp.assets.AssetsPanel` renders the
rows it returns.
"""

# Sort keys shown in the panel's Sort menu, in menu order.
SORT_MANUAL = "manual"
SORT_NAME = "name"
SORT_NAME_DESC = "name_desc"
SORT_NEWEST = "newest"
SORT_OLDEST = "oldest"
SORT_LARGEST = "largest"
SORT_SMALLEST = "smallest"
SORT_SOURCE = "source"

SORTS = [
    (SORT_MANUAL, "Manual order"),
    (SORT_NAME, "Name A\u2013Z"),
    (SORT_NAME_DESC, "Name Z\u2013A"),
    (SORT_NEWEST, "Newest first"),
    (SORT_OLDEST, "Oldest first"),
    (SORT_LARGEST, "Largest first"),
    (SORT_SMALLEST, "Smallest first"),
    (SORT_SOURCE, "Source"),
]

# Filter labels for the source menu. Assets with an empty or unknown source are
# treated as imported, matching what the panel has always displayed.
SOURCE_LABELS = [
    ("generated", "Generated"),
    ("layer", "Layer"),
    ("imported", "Imported"),
]

_SOURCE_RANK = {"generated": 0, "layer": 1, "imported": 2}


def asset_source(asset):
    """The display/filter source for ``asset`` (legacy blanks become imported)."""
    source = getattr(asset, "source", "")
    return source if source in ("generated", "layer") else "imported"


def asset_tags(asset):
    """The asset's tags as a new list."""
    return list(getattr(asset, "tags", None) or [])


def group_name(groups, group_id):
    """The name of ``group_id``, or ``""`` when it is ungrouped or unknown."""
    if group_id is None:
        return ""
    for group in groups or []:
        if group.id == group_id:
            return group.name
    return ""


class AssetViewState:
    """The panel's current search/filter/sort choices.

    ``group_ids`` is a set of :class:`AssetGroup` ids with ``None`` standing for
    the ungrouped pseudo-group; an empty set means "all groups". ``collapsed``
    holds per-session overrides of a group's saved collapse flag.
    """

    def __init__(self):
        self.query = ""
        self.sources = set()
        self.group_ids = set()
        self.tag = None
        self.favorites_only = False
        self.sort = SORT_MANUAL
        self.collapsed = {}

    def has_filters(self):
        """True when any query or filter is narrowing the list."""
        return bool(
            (self.query or "").strip()
            or self.sources
            or self.group_ids
            or self.tag
            or self.favorites_only
        )

    def clear_filters(self):
        """Reset the query and every filter, but keep the sort order."""
        self.query = ""
        self.sources = set()
        self.group_ids = set()
        self.tag = None
        self.favorites_only = False

    def is_collapsed(self, group):
        """Whether ``group`` is collapsed, honouring a session override.

        ``None`` addresses the ungrouped pseudo-group.
        """
        group_id = getattr(group, "id", None) if group is not None else None
        if group_id in self.collapsed:
            return bool(self.collapsed[group_id])
        return bool(getattr(group, "collapsed", False))

    def set_collapsed(self, group, collapsed):
        """Collapse or expand ``group`` for this session (``None`` = ungrouped)."""
        group_id = getattr(group, "id", None) if group is not None else None
        self.collapsed[group_id] = bool(collapsed)

    def toggle_collapsed(self, group):
        """Flip ``group``'s collapse state and return the new state."""
        collapsed = not self.is_collapsed(group)
        self.set_collapsed(group, collapsed)
        return collapsed


def _matches_query(asset, group_label, query):
    if not query:
        return True
    texts = [getattr(asset, "name", "") or "", getattr(asset, "prompt", "") or "",
             group_label or ""]
    texts.extend(asset_tags(asset))
    return any(query in text.casefold() for text in texts)


def filter_assets(assets, groups, state):
    """Return the assets matching ``state``'s query and filters."""
    query = (state.query or "").strip().casefold()
    group_labels = {
        group.id: group.name for group in (groups or [])
    }
    tag = (state.tag or "").casefold()
    result = []
    for asset in assets or []:
        group_id = getattr(asset, "group", None)
        if state.sources and asset_source(asset) not in state.sources:
            continue
        if state.group_ids and group_id not in state.group_ids:
            continue
        if state.favorites_only and not getattr(asset, "favorite", False):
            continue
        if tag and not any(
            tag == candidate.casefold() for candidate in asset_tags(asset)
        ):
            continue
        if not _matches_query(asset, group_labels.get(group_id, ""), query):
            continue
        result.append(asset)
    return result


def sort_assets(assets, sort):
    """Return ``assets`` ordered by ``sort`` (a copy; manual order preserved)."""
    items = list(assets or [])
    if sort == SORT_MANUAL:
        return items
    if sort in (SORT_NAME, SORT_NAME_DESC):
        items.sort(key=lambda a: (a.name or "").casefold(), reverse=sort == SORT_NAME_DESC)
        return items
    if sort == SORT_NEWEST:
        items.sort(key=lambda a: getattr(a, "id", 0), reverse=True)
        return items
    if sort == SORT_OLDEST:
        items.sort(key=lambda a: getattr(a, "id", 0))
        return items
    if sort in (SORT_LARGEST, SORT_SMALLEST):
        def area(asset):
            try:
                width, height = asset.image.size
            except Exception:
                return 0
            return int(width) * int(height)

        items.sort(key=area, reverse=sort == SORT_LARGEST)
        return items
    if sort == SORT_SOURCE:
        items.sort(key=lambda a: (
            _SOURCE_RANK.get(asset_source(a), 2),
            (a.name or "").casefold(),
        ))
        return items
    return items


def build_rows(assets, groups, state):
    """Build the display rows for the current view.

    Returns a list of dicts: ``{"kind": "group", "group", "count", "collapsed"}``
    for a group header and ``{"kind": "asset", "asset", "depth"}`` for an asset.
    With no groups defined the list is flat (no "Ungrouped" header), preserving
    the panel's original look for simple projects.
    """
    filtered = filter_assets(assets, groups, state)
    ordered = sort_assets(filtered, state.sort)
    groups = list(groups or [])

    if not groups:
        return [
            {"kind": "asset", "asset": asset, "depth": 0} for asset in ordered
        ]

    rows = []
    filtering = state.has_filters()
    for group in groups:
        if state.group_ids and group.id not in state.group_ids:
            continue
        members = sort_assets(
            [asset for asset in filtered if asset.group == group.id], state.sort,
        )
        if not members and filtering and group.id not in state.group_ids:
            continue
        collapsed = state.is_collapsed(group)
        rows.append({
            "kind": "group", "group": group, "count": len(members),
            "collapsed": collapsed, "depth": 0,
        })
        if not collapsed:
            rows.extend(
                {"kind": "asset", "asset": asset, "depth": 1}
                for asset in members
            )

    ungrouped = [asset for asset in filtered if asset.group is None]
    show_ungrouped = (
        None in state.group_ids if state.group_ids else bool(ungrouped)
    )
    if show_ungrouped:
        collapsed = state.is_collapsed(None)
        ordered_ungrouped = sort_assets(ungrouped, state.sort)
        rows.append({
            "kind": "group", "group": None, "count": len(ungrouped),
            "collapsed": collapsed, "depth": 0,
        })
        if not collapsed:
            rows.extend(
                {"kind": "asset", "asset": asset, "depth": 1}
                for asset in ordered_ungrouped
            )
    return rows


def all_tags(assets):
    """Every tag used by ``assets``, deduplicated and sorted for a menu."""
    seen = set()
    labels = []
    for asset in assets or []:
        for tag in asset_tags(asset):
            key = tag.casefold()
            if key in seen:
                continue
            seen.add(key)
            labels.append(tag)
    labels.sort(key=str.casefold)
    return labels
