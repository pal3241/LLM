from minillm.data import PackedTokenDataset, clean_text, deterministic_split, document_id


def test_clean_and_hash_are_stable() -> None:
    assert clean_text("halo  \n") == "halo"
    assert document_id("Halo   dunia") == document_id("halo dunia")


def test_packed_dataset_shifts_labels() -> None:
    dataset = PackedTokenDataset(list(range(20)), sequence_length=4)
    inputs, labels = dataset[1]
    assert inputs.tolist() == [4, 5, 6, 7]
    assert labels.tolist() == [5, 6, 7, 8]


def test_split_is_deterministic() -> None:
    records = [{"id": index} for index in range(20)]
    assert deterministic_split(records) == deterministic_split(records)
