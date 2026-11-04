"""Bind selected-bank command evidence to the complete native timeline."""
from collections import Counter

MASKED_COMMANDS = {"ACTMASK", "PREMASK", "MACMASK"}


def annotate_masked_commands(native, masked, data_banks=None):
    """Return new event dictionaries; never change or filter the input timeline.

    A native masked issue must have exactly one physical-mask record and vice
    versa. Optional data_banks validates FFN isolation and each useful MAC mask
    directly from fixed tensor dimensions, independently of producer metadata.
    """
    from staging_audit import require
    fields = ("clock", "command", "Channel", "Rank", "Row", "Column")
    key = lambda e: tuple(e[k] for k in fields)
    expected = Counter(key(e) for e in native if e["command"] in MASKED_COMMANDS)
    actual = Counter(key(e) for e in masked)
    require(expected == actual and all(n == 1 for n in actual.values()),
            "Masked native commands lack exact unique mask evidence")
    masks = {}
    for event in masked:
        mask = event["bank_mask"]
        require(event["command"] in MASKED_COMMANDS and event["Rank"] == 1
                and isinstance(mask, int) and 0 < mask <= 65535,
                "Invalid masked physical-bank command")
        if data_banks is not None:
            require(data_banks in (1, 2), "FFN data-bank count must be one or two")
            banks = [b for b in range(16) if b not in ([0] if data_banks == 1 else [0, 4])]
            compute = sum(1 << b for b in banks)
            require(mask & ~compute == 0, "FFN weight command targets a dedicated data bank")
            if event["command"] == "MACMASK":
                offset = event["Row"]*2048+event["Column"]*32
                base, features = next(((base, features) for base, end, features in (
                    (1048576, 1087488, 256), (1087488, 1126400, 256),
                    (2097152, 2138112, 64)) if base <= offset < end), (None, None))
                require(base is not None, "Masked MAC address outside independent FFN regions")
                columns = (features+len(banks)-1)//len(banks)
                column = ((offset-base)//32)%columns
                useful = sum(1 << bank for ordinal, bank in enumerate(banks)
                             if column*len(banks)+ordinal < features)
                require(mask == useful, "Masked MAC omits useful outputs or computes padding")
        require(event["ready_at"] == event["clock"]+(2 if event["command"] == "ACTMASK" else 1),
                "Masked command duration differs from physical CA encoding")
        masks[key(event)] = mask
    result = []
    for event in native:
        item = dict(event)
        if event["command"] in MASKED_COMMANDS:
            require("bank_mask" not in event or event["bank_mask"] == masks[key(event)],
                    "Native and auxiliary physical bank masks disagree")
            item["bank_mask"] = masks[key(event)]
        result.append(item)
    return result
