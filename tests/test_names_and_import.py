import pytest

from src import ingest, paths, settings
from src.text import name_score, norm


@pytest.mark.parametrize("typed, stored", [
    ("رجب يسري", "رجب يسري"),                 # exact
    ("رجب يسرى", "رجب يسري"),                 # ى instead of ي
    ("حمزه نزار", "حمزة نزار"),               # ه instead of ة
    ("رجب يسر", "رجب يسري"),                  # a letter missing
    ("عبد الكريم نجيب", "عبدالكريم نجيب"),    # typed with a space, stored joined
    ("عبدالكريم نجيب", "عبد الكريم نجيب"),    # and the other way round
    ("رجب", "رجب يسري"),                      # first name only
    ("يسري رجب", "رجب يسري"),                 # words in another order
    ("sarah lee", "sara lee"),                 # works for Latin names too
])
def test_name_score_accepts_real_variants(typed, stored):
    assert name_score(norm(typed), norm(stored)) > 0


@pytest.mark.parametrize("typed, stored", [
    ("ماكسيموس الجلاد", "مامون اسلام"),       # the false positive the old cutoff produced
    ("زيزو الفضائي", "رجب يسري"),
    ("رجب يسري كاكا", "رجب يسري"),            # an extra word that matches nothing
    ("", "رجب يسري"),
    ("رجب", ""),
])
def test_name_score_rejects_strangers(typed, stored):
    assert name_score(norm(typed), norm(stored)) == 0


def test_name_score_never_matches_an_invented_name_in_the_demo_data():
    import openpyxl
    ws = openpyxl.load_workbook(paths.SEED_FILE, read_only=True).active
    names = {norm(r[0]) for r in list(ws.iter_rows(values_only=True))[1:]}
    for fake in ("ماكسيموس الجلاد", "زيزو الفضائي", "هاري بوتر", "كريستيانو رونالدو",
                 "نابليون بونابرت", "سبونج بوب", "جاك سبارو"):
        assert not [n for n in names if name_score(norm(fake), n)], fake


# ---------------------------------------------------------------- import: no model needed

@pytest.fixture
def no_llm(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the model must not be asked for ordinary headers or genders")
    monkeypatch.setattr(ingest, "_llm_json", boom)


ARABIC_HEADER = ("الاسم", "العمر", "النوع", "الجنسية", "المدينة", "الخدمة الرئيسية",
                 "سنوات الخبرة", "التقييم (من 5)", "السعر/الساعة (جنيه)", "رقم الهاتف",
                 "المشاريع المكتملة", "المهارات")


def test_arabic_headers_map_without_a_model(no_llm):
    mapping = ingest._map_columns(ARABIC_HEADER, [])
    assert mapping == {f: i for i, f in enumerate(ingest.FIELDS)}


def test_english_headers_with_spaces_case_and_synonyms(no_llm):
    header = ("Full Name", "City", "Profession", "Hourly Rate", "Mobile", "Sex", "Skills")
    m = ingest._map_columns(header, [])
    assert m == {"name": 0, "city": 1, "service": 2, "hourly_rate": 3, "phone": 4,
                 "gender": 5, "skills": 6}


def test_header_helper_strips_units_and_alternatives():
    assert ingest._header_key("السعر/الساعة (جنيه)") == norm("السعر")
    assert ingest._header_key("  Full  Name ") == "full_name"
    assert ingest._header_key("التقييم (من 5)") == norm("التقييم")


def test_one_column_is_never_claimed_by_two_fields(no_llm):
    m = ingest._map_columns(("الاسم", "اسم", "المدينة", "الخدمة"), [])
    assert m["name"] == 0 and m["city"] == 2 and m["service"] == 3
    assert 1 not in m.values()


def test_unknown_headers_still_fall_back_to_the_model(monkeypatch):
    monkeypatch.setattr(ingest, "_llm_json", lambda *a, **k: {"name": 0, "city": 1, "service": 2})
    assert ingest._map_columns(("w", "x", "y"), [("a", "b", "c")]) == {"name": 0, "city": 1, "service": 2}


def test_gender_words_in_arabic_and_english(monkeypatch):
    monkeypatch.setattr(ingest, "_llm_json", lambda *a, **k: None)
    got = ingest._gender_map(["ذكر", "أنثى", "انثى", "Male", "FEMALE", "F", "m", "ست", "راجل", "???"])
    assert got["ذكر"] == got["Male"] == got["m"] == got["راجل"] == "male"
    assert got["أنثى"] == got["انثى"] == got["FEMALE"] == got["F"] == got["ست"] == "female"
    assert "???" not in got


def test_demo_workbook_imports_offline(no_llm):
    records, report = ingest.parse_excel(paths.SEED_FILE.read_bytes())
    assert len(records) == 1000 and report["invalid_rows"] == 0
    genders = [r["gender"] for r in records]
    assert genders.count("male") == 700 and genders.count("female") == 300
    first = records[0]
    assert first["name"] == "حمزة نزار" and first["city"] == "العدوة" and first["rating"] == 5.0
    assert first["hourly_rate"] == 180 and first["phone"] == "01144173705"


# ---------------------------------------------------------------- environment knobs

def test_env_float_reads_validates_and_defaults(monkeypatch):
    monkeypatch.delenv("X_KNOB", raising=False)
    assert settings._env_float("X_KNOB", 0.5) == 0.5
    monkeypatch.setenv("X_KNOB", " 0.25 ")
    assert settings._env_float("X_KNOB", 0.5) == 0.25
    monkeypatch.setenv("X_KNOB", "")
    assert settings._env_float("X_KNOB", 0.5) == 0.5
    monkeypatch.setenv("X_KNOB", "lots")
    with pytest.raises(ValueError, match="must be a number"):
        settings._env_float("X_KNOB", 0.5)
    monkeypatch.setenv("X_KNOB", "7")
    with pytest.raises(ValueError, match="between"):
        settings._env_float("X_KNOB", 0.5)


def test_defaults_of_the_two_search_knobs():
    assert settings.RANK_SIM_WEIGHT == 0.35 and settings.NAME_SIM_MIN == 0.90
