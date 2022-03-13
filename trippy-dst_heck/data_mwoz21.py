import json
import re
from utils import DSTExample, convert_to_unicode
from tqdm import tqdm

# `delexicalize_utterance` function in this code is modified on March 10th, 2022 by squiduu.

# for mapping slot names in dialogue_acts.json file to proper designations
ACT_DICT = {
    "taxi-depart": "taxi-departure",
    "taxi-dest": "taxi-destination",
    "taxi-leave": "taxi-leaveAt",
    "taxi-arrive": "taxi-arriveBy",
    "train-depart": "train-departure",
    "train-dest": "train-destination",
    "train-leave": "train-leaveAt",
    "train-arrive": "train-arriveBy",
    "train-people": "train-book_people",
    "restaurant-price": "restaurant-pricerange",
    "restaurant-people": "restaurant-book_people",
    "restaurant-day": "restaurant-book_day",
    "restaurant-time": "restaurant-book_time",
    "hotel-price": "hotel-pricerange",
    "hotel-people": "hotel-book_people",
    "hotel-day": "hotel-book_day",
    "hotel-stay": "hotel-book_stay",
    "booking-people": "booking-book_people",
    "booking-day": "booking-book_day",
    "booking-stay": "booking-book_stay",
    "booking-time": "booking-book_time",
}

# loaded from data config file
LABEL_MAPS = {}


def load_acts(act_file: str):
    """Load the `dialogue_acts.json` and returns a list of slotvalue pairs.

    Args:
        act_file (str): an input dialogue acts.json file

    Returns:
        action_dict (dict): {(diag_ids.json, diag_turns, slot): [slotvalue]}
    """
    with open(file=act_file, mode="r") as f:
        # acts: {diag_ids: {diag_turns: {domain-act: [[slotnames, slotvalues]]}}}
        acts = json.load(f)

    action_dict = {}
    # dial_ids (str): 'PMUL3994', 'PMUL3995', ...
    for dial_ids in acts:
        # dial_turns (str): '1', '2', ...
        for dial_turns in acts[dial_ids]:
            # preprocess only if turn has annotation
            if isinstance(acts[dial_ids][dial_turns], dict):
                # act (str): 'Attraction-Request', 'Attraction-Inform', ...
                for act in acts[dial_ids][dial_turns]:
                    temp_act = act.lower()
                    temp_act = temp_act.split("-")
                    if (
                        temp_act[1] == "inform"
                        or temp_act[1] == "recommend"
                        or temp_act[1] == "select"
                        or temp_act[1] == "book"
                    ):
                        # sv_pair_list (list): [slotname, slotvalue]
                        for sv_pair_list in acts[dial_ids][dial_turns][act]:
                            # slotnames (str)
                            slotname = sv_pair_list[0].lower()
                            slotvalue = sv_pair_list[1].lower()
                            # slotvalues (str)
                            slotvalue = sv_pair_list[1].strip()
                            if slotname == "none" or slotvalue == "?" or slotvalue == "none":
                                continue
                            # slot (str): 'domain-slotname'
                            slot = temp_act[0] + "-" + slotname

                            if slot in ACT_DICT:
                                # i.e., 'taxi-depart' -> 'taxi-departure'
                                slot = ACT_DICT[slot]

                            key = dial_ids + ".json", dial_turns, slot

                            if key not in action_dict:
                                action_dict[key] = list([slotvalue])

    return action_dict


def tokenize(uttr: str):
    """Convert input utterances to a list of tokenized input utterances.

    Args:
        uttr (str): an input utterance.

    Returns:
        uttr_toks (list): a list of tokenized input utterances.
    """
    # convert uttr to str if it is bytes
    uttr = convert_to_unicode(uttr)
    uttr = uttr.lower()
    uttr_toks = [tok for tok in map(str.strip, re.split(pattern="(\W+)", string=uttr)) if len(tok) > 0]

    return uttr_toks


def delexicalize_utterance(uttr_list: str, inform_dict: dict, unk_token: str = "[UNK]"):
    """Get delexicalized utterance that slotvalues are replaced by `[UNK]` token.
    This function is modified from the corresponding function of `TripPy`.

    Args:
        uttr (list): an input utterance.
        delex (dict): {slot: [slotvalue]}, slotvalues are from `turn_label`.
        unk_token (str, optional): an unknown vocab token. Defaults to `[UNK]`.

    Returns:
        uttr (list): a tokenized utterance that slotvalues are replaced by `[UNK]`
    """
    # uttr_list (list): [str, str, str, ...]
    uttr_list = tokenize(uttr_list)

    for _, values in inform_dict.items():
        for value in values:
            if values != "none":
                value = tokenize(value)
                for i in range(len(uttr_list) + 1 - len(value)):
                    if uttr_list[i : i + len(value)] == value:
                        uttr_list[i : i + len(value)] = [unk_token] * len(value)

    return uttr_list


def normalize_time(text: str):
    """Normalize text of time expressions.

    Args:
        text (str): an input text of time expressions to normalize.

    Returns:
        text (str): a normalized text of time expressions.
    """
    # 10am or 10pm -> 10 am or 10 pm
    text = re.sub(pattern="(\d{1})(a\.?m\.?|p\.?m\.?)", repl=r"\1 \2", string=text)
    # 10 am or 10 pm -> 10:00 am or 10:00 pm
    text = re.sub(pattern="(^| )(\d{1,2}) (a\.?m\.?|p\.?m\.?)", repl=r"\1\2:00 \3", string=text)
    # correct missing separator
    text = re.sub(
        pattern="(^| )(at|from|by|until|after) ?(\d{1,2}) ?(\d{2})([^0-9]|$)", repl=r"\1\2 \3:\4\5", string=text,
    )
    # correct wrong separator
    text = re.sub(pattern="(^| )(\d{2})[;.,](\d{2})", repl=r"\1\2:\3", string=text)
    # normalize simple full hour time
    text = re.sub(pattern="(^| )(at|from|by|until|after) ?(\d{1,2})([;., ]|$)", repl=r"\1\2 \3:00\4", string=text,)
    # add missing leading 0
    text = re.sub(pattern="(^| )(\d{1}:\d{2})", repl=r"\g<1>0\2", string=text)
    # map 12 hour times to 24 hour times
    text = re.sub(
        pattern="(\d{2})(:\d{2}) ?p\.?m\.?",
        repl=lambda x: str(int(x.groups()[0]) + 12 if int(x.groups()[0]) < 12 else int(x.groups()[0])) + x.groups()[1],
        string=text,
    )
    # correct times that use 24 as hour
    text = re.sub(pattern="(^| )24:(\d{2})", repl=r"\g<1>00:\2", string=text)

    return text


def normalize_label(domain_slotname: str, slotvalue_label: str):
    """Normalize the labels as well as `LABEL_MAPS` dictionary.

    Args:
        slot (str): a string of `domain`-`slotname`.
        slotvalue_label (str): a gold slotvalue label corresponding to the `domain`-`slotname`.

    Returns:
        slotvalue_label (str): a normalized gold slotvalue label to clear classification.
    """
    # normalize the empty slotvalue labels
    if slotvalue_label == "" or slotvalue_label == "not_mentioned":
        return "none"

    # normalize the time slotvalue labels
    if "leaveAt" in domain_slotname or "arriveBy" in domain_slotname or domain_slotname == "restaurant-book_time":
        return normalize_time(slotvalue_label)

    # map slotvalue label: guesthouse -> guest house
    if (
        "type" in domain_slotname
        or "name" in domain_slotname
        or "destination" in domain_slotname
        or "departure" in domain_slotname
    ):
        slotvalue_label = re.sub(pattern="guesthouse", repl="guest house", string=slotvalue_label)

    # map to boolean slotvalues
    if domain_slotname == "hotel-parking" or domain_slotname == "hotel-internet":
        if slotvalue_label == "yes" or slotvalue_label == "free":
            return "true"
        if slotvalue_label == "no":
            return "false"
    if domain_slotname == "hotel-type":
        if slotvalue_label == "hotel":
            return "true"
        if slotvalue_label == "guest house":
            return "false"

    return slotvalue_label


def get_token_position(toks: list, v_label: str):
    """Get the indices of position of the slotvalue label in utterance token list.

    Args:
        toks (list): a list of tokens e.g., user utterance tokens.
        v_label (str): slot-name or none or dontcare.

    Returns:
        existence (bool): whether it exists or not.
        find_pos (list): a list of label position index in tokens.
    """
    find_pos = []
    existence = False
    labels_list = [item for item in map(str.strip, re.split(pattern="(\W+)", string=v_label)) if len(item) > 0]
    for i in range(len(toks) + 1 - len(labels_list)):
        if toks[i : i + len(labels_list)] == labels_list:
            find_pos.append((i, i + len(labels_list)))
            existence = True

    return existence, find_pos


def check_label_existence(v_label: str, usr_uttr_toks: list):
    """Check that `label` is in utterance, and get the position if it exists.

    Args:
        v_label (str): a slotvalue in `turn_label` or a slotvalue in `LABEL_MAPS`.
        usr_uttr_toks_list (list): a tokenized user utterance.

    Returns:
        isin_usr_uttr (bool): whether there is the label in user utterance or not.
        label_pos_in_usr_uttr_list (list): label position indices in tokens.
    """
    isin_uttr, label_pos_in_uttr = get_token_position(toks=usr_uttr_toks, v_label=v_label)
    if not isin_uttr and v_label in LABEL_MAPS:
        # v_label_eqv (str): a slotvalue equivalent to the slotvalue label
        for v_label_eqv in LABEL_MAPS[v_label]:
            isin_uttr, label_pos_in_uttr = get_token_position(toks=usr_uttr_toks, v_label=v_label_eqv)
            if isin_uttr:
                break

    return isin_uttr, label_pos_in_uttr


def isin_list(tok: str, v: str):
    """Check if there is `value` in list of `tok`.

    Args:
        tok (str): a str to check with a form of list.
        v (str): a str to find in list of `tok`.

    Returns:
        existence (bool): whether `value` is in list of `tok` or not.
    """
    existence = False
    toks = [item for item in map(str.strip, re.split(pattern="(\W+)", string=tok)) if len(item) > 0]
    vs = [item for item in map(str.strip, re.split(pattern="(\W+)", string=v)) if len(item) > 0]
    for i in range(len(toks) + 1 - len(vs)):
        if toks[i : i + len(vs)] == vs:
            existence = True
            break

    return existence


def check_slot_inform(v_label: str, inform_labels: list):
    """Fuzzy matching (a string similarity search algorithm) to label informed slotvalues.

    Args:
        v_label (str): a gold slotvalue to check.
        inform_labels (list):

    Returns:
        match (bool): whether they match each other or not.
        informed_v (str): a matched informed label if `match` equals to `True`.
    """
    match = False
    informed_v = "none"
    # v_label (str): tokenized slotvalue label including spacing between tokens
    v_label = " ".join(tokenize(v_label))

    # inform_label (str): an informed slotvalue
    for inform_label in inform_labels:
        if v_label == inform_label:
            match = True
        elif isin_list(tok=inform_label, v=v_label):
            match = True
        elif isin_list(tok=v_label, v=inform_label):
            match = True

        elif inform_label in LABEL_MAPS:
            for inform_label_eqv in LABEL_MAPS[inform_label]:
                if v_label == inform_label_eqv:
                    match = True
                    break
                elif isin_list(tok=inform_label_eqv, v=v_label):
                    match = True
                    break
                elif isin_list(tok=v_label, v=inform_label_eqv):
                    match = True
                    break

        elif v_label in LABEL_MAPS:
            for v_label_eqv in LABEL_MAPS[v_label]:
                if v_label_eqv == inform_label:
                    match = True
                    break
                elif isin_list(tok=inform_label, v=v_label_eqv):
                    match = True
                    break
                elif isin_list(tok=v_label_eqv, v=inform_label):
                    match = True
                    break

        if match:
            informed_v = inform_label
            break

    return match, informed_v


def check_slot_referral(v_label: str, s: str, seen_s: dict):
    """Check whether `seen slot` is gold slotvalue or not, and return `seen slot` if it is gold.

    Args:
        v_label (str): a gold slotvalue label.
        s (str): a slotname including `domain`.
        seen_s (dict): a seen slotname which appeared earlier.

    Returns:
        referred_s (str): a slot if seen slotvalue equals gold.
    """
    referred_s = "none"

    if s in ["hotel-stars", "hotel-internet", "hotel-parking"]:
        return referred_s

    for ss in seen_s:
        if ss in ["hotel-stars", "hotel-internet", "hotel-parking"]:
            continue
        if re.match(pattern="(hotel|restaurant)-book_people", string=s) and ss == "hotel-book_day":
            continue
        if re.match(pattern="(hotel|restaurant)-book_people", string=ss) and s == "hotel-book_day":
            continue
        if s != ss and s not in seen_s or seen_s[s] != v_label:
            if seen_s[s] == v_label:
                referred_s = s
                break
            elif v_label in LABEL_MAPS:
                for v_label_eqv in LABEL_MAPS[v_label]:
                    if seen_s[s] == v_label_eqv:
                        referred_s = s
                        break

    return referred_s


def get_turn_label(
    v_label: str, inform_labels: list, usr_uttr_toks: list, s: str, seen_s: dict, s_last_occurrence: bool,
):
    """Get gold variables to learn span, inform, refer, and gate for each turn.

    Args:
        v_label (str): a gold slot-value to span e.g., 'taxi-arriveBy'
        inform_labels (list): a gold slotvalue to inform from sys.
        sys_uttr_toks (list): a tokenized system utterances.
        usr_uttr_toks (list): a tokenized user utterances.
        s (str): a string of `slot`.
        seen_s (dict): {`slotname`: `slotvalue`} pairs that came out before.
        s_last_occurrence (bool): whether the `slot` appears just before or not.

    Returns:
        informed_v (str): a matched gold `slotvalue` when `slotgate` equals inform.
        referred_s (str): a matched gold 'slotname' when 'slotgate' equals refer.
        usr_uttr_toks_label (list): only gold tokens are 1, otherwise is 0 for span.
        g (str): a gate that determines state of each slot.
    """
    # set variables in advance
    usr_uttr_toks_label = [0 for _ in usr_uttr_toks]
    informed_v = "none"
    referred_s = "none"

    if v_label == "none" or v_label == "dontcare" or v_label == "true" or v_label == "false":
        g = v_label
    else:
        isin_uttr, label_pos_in_uttr = check_label_existence(v_label=v_label, usr_uttr_toks=usr_uttr_toks)
        is_informed, informed_v = check_slot_inform(v_label=v_label, inform_labels=inform_labels)
        if isin_uttr:
            g = "copy_value"
            if s_last_occurrence:
                start_pos, end_pos = label_pos_in_uttr[-1]
                for i in range(start_pos, end_pos):
                    usr_uttr_toks_label[i] = 1
            else:
                for start_pos, end_pos in label_pos_in_uttr:
                    for i in range(start_pos, end_pos):
                        usr_uttr_toks_label[i] = 1
        elif is_informed:
            g = "inform"
        else:
            referred_s = check_slot_referral(v_label=v_label, s=s, seen_s=seen_s)
            if referred_s != "none":
                g = "refer"
            else:
                g = "unpointable"

    return informed_v, referred_s, usr_uttr_toks_label, g


def create_examples(
    input_file: str,
    act_file: str,
    run_type: str,
    slot_list: list,
    label_maps: dict = {},
    append_hst: bool = False,
    label_hst: bool = False,
    swap_uttr: bool = False,
    label_mentioned_v: bool = False,
    delex_sys_uttr: bool = False,
    unk_token: str = "[UNK]",
):
    """Read a DST json file into a list of `DSTExamples`.

    Args:
        input_file (str): path of data file.
        acts_file (str): path of dialogue action file.
        run_type (str): one of `train`, `valid`, `test`.
        slot_list (list): a list of all slot-names.
        label_maps (dict, optional): equivalent words of slot-values in data config file. Defaults to {}.
        append_hst (bool, optional): whether or not to append history to each turn.
        label_hst (bool, optional): whether or not to label history as well.
        swap_uttr (bool, optional): whether or not to swap turn utterances (default: sys|usr, swapped: usr|sys).
        label_mentioned_v (bool, optional): whether or not to label values that have been mentioned before.
        delex_sys_uttr (bool, optional): whether or not to delexicalize system utterances.
    """

    # sys_inform_dict (dict): {(dial_ids.json, dial_turns, slot): [slotvalue]}
    sys_inform_dict = load_acts(act_file)

    # read dialogue data
    with open(file=input_file, mode="r", encoding="utf-8") as f:
        rawdata = json.load(f)

    global LABEL_MAPS
    LABEL_MAPS = label_maps

    examples = []
    # dial_ids: dialogue ids i,e., 'MUL0484.json'
    for dial_ids in tqdm(rawdata):
        log_data = rawdata[dial_ids]["log"]

        # collect all slot changes throughout the dialogues
        cumulative_labels = {s: "none" for s in slot_list}
        # sys uttr is empty at first, since mwoz starts with user uttr
        uttr_toks_list = [[]]
        modified_sv_list = [{}]
        # collect all uttrs and their metadata
        switch_speaker = True

        dial_turn = 0

        # uttr_metadata_dict: {'text': uttr, 'metadata': metadata}
        for uttr_metadata_dict in log_data:
            # assert sys and usr uttr alternate
            is_sys_uttr = uttr_metadata_dict["metadata"] != {}
            if switch_speaker == is_sys_uttr:
                print("WARN: Wrong order of sys and usr uttrs, skipping rest of dials %s" % (dial_ids))
                break
            switch_speaker = is_sys_uttr

            if is_sys_uttr:
                dial_turn += 1

            # delexicalize sys uttr
            if delex_sys_uttr and is_sys_uttr:
                # inform_dict (dict): {slot: 'none'}, ...
                inform_dict = {s: "none" for s in slot_list}
                # slot (str): `domain-slotname`
                for s in slot_list:
                    if (str(dial_ids), str(dial_turn), s,) in sys_inform_dict:
                        # inform_dict (dict): {slot: [slotvalue]}
                        inform_dict[s] = sys_inform_dict[(str(dial_ids), str(dial_turn), s)]
                # normalize uttr with delexicalization
                # uttr_toks_list (list): [[tokenized_delex_uttr], [tokenized_delex_uttr], ..., [tokenized_delex_uttr]]
                uttr_toks_list.append(
                    delexicalize_utterance(
                        uttr_list=uttr_metadata_dict["text"], inform_dict=inform_dict, unk_token=unk_token,
                    )
                )
            else:
                # normalize uttr
                uttr_toks_list.append(tokenize(uttr_metadata_dict["text"]))

            modified_s_dict = {}

            # extract metadata if it is sys uttr
            if is_sys_uttr:
                # domain (str): domain
                for d in uttr_metadata_dict["metadata"]:
                    booked = uttr_metadata_dict["metadata"][d]["book"]["booked"]
                    booked_s_dict = {}

                    # check booked section
                    if booked != []:
                        # s (str): slotname
                        for s in booked[0]:
                            # booked_slots_dict (dict): {slotname: normalized slotvalue}
                            booked_s_dict[s] = normalize_label(domain_slotname=f"{d}-{s}", slotvalue_label=booked[0][s])

                    for cat in ["book", "semi"]:
                        # s (str): slotname
                        for s in uttr_metadata_dict["metadata"][d][cat]:
                            cat_s = f"{d}-book_{s}" if cat == "book" else f"{d}-{s}"
                            # get normalized gold slotvalues
                            v_label = normalize_label(
                                domain_slotname=cat_s, slotvalue_label=uttr_metadata_dict["metadata"][d][cat][s],
                            )
                            # prefer slotvalues as stored in booked section
                            if s in booked_s_dict:
                                v_label = booked_s_dict[s]
                            if cat_s in slot_list and cumulative_labels[cat_s] != v_label:
                                modified_s_dict[cat_s] = v_label
                                cumulative_labels[cat_s] = v_label

            modified_sv_list.append(modified_s_dict.copy())

        # basic proper form: (usr, sys) turns
        # set variables in advance
        dial_turn = 0
        dial_seen_s_dict = {}
        dial_seen_sv_dict = {s: "none" for s in slot_list}
        ds_dict = {slot: "none" for slot in slot_list}
        sys_uttr_toks_list = []
        usr_uttr_toks_list = []
        hst_toks_list = []
        hst_toks_labels_dict = {s: [] for s in slot_list}

        for i in range(1, len(uttr_toks_list) - 1, 2):
            sys_uttr_toks_labels_dict = {}
            usr_uttr_toks_labels_dict = {}
            sv_labels_dict = {}
            inform_dict = {}
            inform_s_dict = {}
            referral_dict = {}
            g_dict = {}

            # collect turn data
            if append_hst:
                if swap_uttr:
                    hst_toks_list = usr_uttr_toks_list + sys_uttr_toks_list + hst_toks_list
                else:
                    hst_toks_list = sys_uttr_toks_list + usr_uttr_toks_list + hst_toks_list

            # order of utterances: sys -> usr -> sys -> ...
            # uttr_toks_list[0] is empty since mwoz starts with user utterance
            sys_uttr_toks_list = uttr_toks_list[i - 1]
            usr_uttr_toks_list = uttr_toks_list[i]
            # turn_sv_dict (dict): {domain-slotname: slotvalue}
            turn_sv_dict = modified_sv_list[i + 1]

            # set globally unique identifier (GUID) for dialogues e.g., 'train-800-0'
            guid = f"{run_type}-{str(dial_ids)}-{str(dial_turn)}"

            temp_hst_toks_labels_dict = hst_toks_labels_dict.copy()
            temp_ds_dict = ds_dict.copy()
            # s (str): 'domain-slotname'
            for s in slot_list:
                v_label = "none"
                if s in turn_sv_dict:
                    v_label = turn_sv_dict[s]
                    # sv_labels[s] (str): 'none' if it does not exist
                    sv_labels_dict[s] = v_label
                elif label_mentioned_v and s in dial_seen_s_dict:
                    v_label = dial_seen_sv_dict[s]

                # get dialogue action annotations
                inform_labels_list = ["none"]
                if (str(dial_ids), str(dial_turn), s) in sys_inform_dict:
                    # inform_labels_list (list): ['slotvalue']
                    inform_labels_list = [
                        normalize_label(domain_slotname=s, slotvalue_label=i)
                        for i in sys_inform_dict[(str(dial_ids), str(dial_turn), s)]
                    ]
                elif (str(dial_ids), str(dial_turn), "booking-" + s.split("-")[1]) in sys_inform_dict:
                    inform_labels_list = [
                        normalize_label(domain_slotname=s, slotvalue_label=i)
                        for i in sys_inform_dict[(str(dial_ids), str(dial_turn), "booking-" + s.split("-")[1])]
                    ]

                # get gold labels for inform, refer, span, gate
                informed_v, referred_s, usr_uttr_toks_label, g = get_turn_label(
                    v_label=v_label,
                    inform_labels=inform_labels_list,
                    usr_uttr_toks=usr_uttr_toks_list,
                    s=s,
                    seen_s=dial_seen_sv_dict,
                    s_last_occurrence=True,
                )
                inform_dict[s] = informed_v

                # inform_s_dict (dict): {domain-slotname: 0 or 1} - is there any inform in the slot?
                if informed_v != "none":
                    inform_s_dict[s] = 1
                else:
                    inform_s_dict[s] = 0

                # do not use span prediction on sys uttr, use inform prediction instead
                sys_uttr_toks_label = [0 for _ in sys_uttr_toks_list]

                # determine what to do with slotvalue repetitions
                # tag slotvalue if it is unique in seen slots, otherwise not
                # since correct slot assignment cannot be guaranteed anymore
                if label_mentioned_v and s in dial_seen_s_dict:
                    if g == "copy_value" and list(dial_seen_sv_dict.values()).count(v_label) > 1:
                        g = "none"
                        usr_uttr_toks_label = [0 for _ in usr_uttr_toks_label]

                sys_uttr_toks_labels_dict[s] = sys_uttr_toks_label
                usr_uttr_toks_labels_dict[s] = usr_uttr_toks_label

                if append_hst:
                    if label_hst:
                        if swap_uttr:
                            temp_hst_toks_labels_dict[s] = (
                                usr_uttr_toks_label + sys_uttr_toks_label + temp_hst_toks_labels_dict[s]
                            )
                        else:
                            temp_hst_toks_labels_dict[s] = (
                                sys_uttr_toks_label + usr_uttr_toks_label + temp_hst_toks_labels_dict[s]
                            )
                    else:
                        temp_hst_toks_labels_dict[s] = [
                            0 for _ in sys_uttr_toks_label + usr_uttr_toks_label + temp_hst_toks_labels_dict[s]
                        ]

                # map all occurances of unpointable slotvalues to none
                # however, since the labels will still suggest a presence of unpointable slot values
                # it is just not possible to do that via span prediction on the current input
                if g == "unpointable":
                    g_dict[s] = "none"
                    referral_dict[s] = "none"
                elif s in dial_seen_s_dict and g == dial_seen_s_dict[s] and g != "copy_value" and g != "inform":
                    g_dict[s] = "none"
                    referral_dict[s] = "none"
                else:
                    g_dict[s] = g
                    referral_dict[s] = referred_s

                # remember that this slot was mentioned during this dialogue already
                if g != "none":
                    dial_seen_s_dict[s] = g
                    dial_seen_sv_dict[s] = v_label
                    temp_ds_dict[s] = g
                    # unpointable is not a valid gate, therefore replace with some valid gate for now
                    if g == "unpointable":
                        temp_ds_dict[s] = "copy_value"

            # make input data after creating golden slotvalue pairs
            if swap_uttr:
                # uttr_a, b (list): [tok, tok, ..., tok]
                uttr_a = usr_uttr_toks_list
                uttr_b = sys_uttr_toks_list

                # uttr_a, b_labels (dict): {domain-slotname: [0, 1, 1, ..., 0]}
                uttr_a_labels = usr_uttr_toks_labels_dict
                uttr_b_labels = sys_uttr_toks_labels_dict
            else:
                uttr_a = sys_uttr_toks_list
                uttr_b = usr_uttr_toks_list

                uttr_a_labels = sys_uttr_toks_labels_dict
                uttr_b_labels = usr_uttr_toks_labels_dict

            # create input sequences and gold variables for DST
            examples.append(
                DSTExample(
                    guid=guid,
                    uttr_a=uttr_a,
                    uttr_b=uttr_b,
                    hst_toks_list=hst_toks_list,
                    uttr_a_labels=uttr_a_labels,
                    uttr_b_labels=uttr_b_labels,
                    hst_toks_labels_dict=hst_toks_labels_dict,
                    dial_seen_sv_dict=dial_seen_sv_dict.copy(),
                    inform_dict=inform_dict,
                    inform_s_dict=inform_s_dict,
                    referral_dict=referral_dict,
                    ds_dict=ds_dict,
                    g_dict=g_dict,
                )
            )

            # update history labels and dialogue states
            hst_toks_labels_dict = temp_hst_toks_labels_dict.copy()
            ds_dict = temp_ds_dict.copy()

            dial_turn += 1

    return examples
