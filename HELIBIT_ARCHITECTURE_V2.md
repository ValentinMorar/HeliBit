# HeliBit-AI v2 — Arhitectură de refactorizare
## De la "clasificator peste etichete memorate" la "motor compozițional real"

---

## 0. Diagnostic (rezumat, pentru context Claude Code)

Implementarea actuală (`engine.py::predict`) face **nearest-neighbor classification** peste
un set fix de string-uri (`known_targets`), populat exclusiv din coloana de target a
datelor de antrenament. Scorul e o combinație de:
- frecvență de bigrame token→target (`co_occurrence_weights`),
- suprapunere/hamming distance între bitboard-ul compus al input-ului și bitboard-ul
  fiecărui target candidat.

**Consecință structurală**: sistemul nu poate produce niciodată un output care nu a
apărut *deja, cuvânt cu cuvânt, ca target* în antrenament. Nu există nicio operație de
adunare/scădere reală (ex: "quarter" = +15 min) implementată — traiectoriile elicoidale
și Lyapunov Guard din propunere există în cod, dar nu participă efectiv la calculul
răspunsului (`traj_collapse` e doar un termen mic de penalizare în scor).

**Scopul v2**: sistemul trebuie să **extragă roluri semantice din input** (oră de bază,
offset de minute, direcție, meridian) și să **calculeze** rezultatul din acele roluri,
nu să-l caute într-un tabel. Bitboard-urile, traiectoriile și dinamica Hebbian trebuie
repoziționate ca mecanisme reale de reprezentare/disambiguare, nu ca decor peste un
lookup.

---

## 1. Principiul arhitectural nou

```
INPUT TEXT
   │
   ▼
[1] Tokenizare + encodare bitboard PE ROL (nu pe identitate de cuvânt)
   │
   ▼
[2] Extragere de SLOTURI semantice (bitmask matching, nu string matching)
      slots = { hour: int|None, minute_offset: int|None,
                direction: {PAST, TO, EXACT}|None, meridiem: {AM,PM}|None }
   │
   ▼
[3] Rezolvare / disambiguare sloturi incomplete sau ambigue
      (aici intră traiectoria elicoidală + Hebbian — vezi §5)
   │
   ▼
[4] MOTOR ARITMETIC (integer, zero-FPU — respectă spiritul propunerii)
      HH:MM = f(hour, minute_offset, direction, meridiem)
   │
   ▼
[5] Validare / abstenție — dacă sloturile esențiale lipsesc, sistemul
    răspunde EXPLICIT "necunoscut", nu completează cu o valoare implicită
   │
   ▼
OUTPUT: "HH:MM"  sau  UNKNOWN
```

Diferența esențială față de v1: **targetul nu mai e ales dintr-o listă închisă** —
e **construit** din sloturi. Asta elimină structural problema "nu poate genera
04:10 dacă n-a văzut-o exact la antrenament".

---

## 2. Reprezentare bitboard nouă (rol explicit, nu hash opac)

**Problema actuală** (`bitboard.py::from_string`, de verificat, dar reiese din
comportament): fiecare token primește un bitboard derivat dintr-un hash al
string-ului — o amprentă opacă, fără structură internă. De-aia Hamming distance
între "four" și "5" nu înseamnă nimic aritmetic.

**Reproiectare propusă**: bitboard-ul de 64 de biți e împărțit în **câmpuri cu sens**:

| Biți     | Câmp                  | Conținut                                                   |
|----------|-----------------------|-------------------------------------------------------------|
| 63–60    | `ROLE`                | 0=NUMBER, 1=UNIT_HOUR("o'clock"), 2=DIRECTION_PAST, 3=DIRECTION_TO, 4=MODIFIER_QUARTER, 5=MODIFIER_HALF, 6=MERIDIEM_AM, 7=MERIDIEM_PM, 8=UNKNOWN |
| 59–54    | `VALUE`               | valoare numerică binară (0–63), folosită doar dacă ROLE=NUMBER |
| 53–0     | `LEXICAL_FINGERPRINT` | amprenta actuală (hash), păstrată pentru compatibilitate cu resonance_score / POPCNT, dar acum e *secundară*, nu sursa de adevăr |

Asta permite operații reale, nu doar comparații de similaritate:
- `AND` cu o mască de rol → filtrare instant a tipului de token (SIMD, cum promite propunerea).
- `SHR 60` → extragere directă a rolului.
- `(board >> 54) & 0x3F` → extragere directă a valorii numerice, dacă ROLE=NUMBER.

Fișier de atins: `helibit/bitboard.py`. Adaugă:
```python
class Bitboard64:
    ROLE_SHIFT = 60
    VALUE_SHIFT = 54
    VALUE_MASK = 0x3F

    @classmethod
    def from_role_value(cls, role: int, value: int = 0, fingerprint: int = 0) -> "Bitboard64":
        v = (role << cls.ROLE_SHIFT) | ((value & cls.VALUE_MASK) << cls.VALUE_SHIFT) | (fingerprint & ((1 << 54) - 1))
        return cls(v)

    def role(self) -> int:
        return (self.value >> self.ROLE_SHIFT) & 0xF

    def numeric_value(self) -> int:
        return (self.value >> self.VALUE_SHIFT) & self.VALUE_MASK
```

---

## 3. Modul nou: `lexicon.py` — dicționar de roluri (înlocuiește hash-ul orb)

Un dicționar mic, explicit, scris o singură dată (nu învățat — e gramatică închisă,
exact ca-n propunere: "closed, deterministic sequence tasks"):

```python
ROLE_NUMBER, ROLE_HOUR_UNIT, ROLE_PAST, ROLE_TO, ROLE_QUARTER, ROLE_HALF, ROLE_AM, ROLE_PM = range(8)

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "fifteen": 15, "twenty": 20, "twenty-five": 25, "thirty": 30, "forty": 40,
    # ... extensibil, dar EXPLICIT, nu memorat din perechi (text, timp)
}
DIRECTION_WORDS = {"past": ROLE_PAST, "to": ROLE_TO}
MODIFIER_WORDS = {"quarter": ROLE_QUARTER, "half": ROLE_HALF}
MERIDIEM_WORDS = {"am": ROLE_AM, "pm": ROLE_PM}
```

**De ce e diferit de v1, fundamental**: `co_occurrence_weights` din v1 învață
*asocieri statistice* (cât de des apare "quarter" lângă targetul "04:15"). Aici,
"quarter" înseamnă *întotdeauna* 15 minute, ca fapt lingvistic fix — exact cum
gramatica orelor e într-adevăr o gramatică formală închisă, nu ceva ce trebuie
"învățat" din exemple. Partea care CHIAR trebuie învățată/adaptivă (vezi §5) e
robustețea la formulări noi, nu regulile de bază.

---

## 4. Motorul aritmetic (`arithmetic.py`, nou)

```python
def resolve_time(slots: dict) -> Optional[str]:
    """
    slots = {
        "hour": int | None,           # 1-12
        "minute_offset": int | None,  # 0-59
        "direction": "PAST"|"TO"|"EXACT"|None,
        "meridiem": "AM"|"PM"|None,
    }
    Returns "HH:MM" (24h) sau None dacă sloturile esențiale lipsesc.
    """
    if slots["hour"] is None:
        return None  # ABSTENȚIE — nu ghici

    hour = slots["hour"] % 12
    minute = 0
    direction = slots["direction"] or "EXACT"

    if direction == "EXACT":
        minute = slots["minute_offset"] or 0
    elif direction == "PAST":
        minute = slots["minute_offset"] or 0
    elif direction == "TO":
        offset = slots["minute_offset"] or 0
        minute = (60 - offset) % 60
        if offset != 0:
            hour = (hour - 1) % 12   # "quarter TO five" -> ora devine 4, nu 5

    # meridiem -> 24h
    if slots["meridiem"] == "PM" and hour != 12:
        hour += 12
    elif slots["meridiem"] == "PM" and hour == 12:
        hour = 12
    elif slots["meridiem"] == "AM" and hour == 12:
        hour = 0
    elif slots["meridiem"] is None and hour == 0:
        hour = 12  # fallback afișare 12h dacă nu avem meridiem

    display_hour = hour if hour != 0 else 12
    return f"{display_hour:02d}:{minute:02d}"
```

Asta e chiar tipul de calcul integer, fără FPU, pe care propunerea originală îl
promitea — doar că acum chiar există, în loc să fie înlocuit de un lookup.

---

## 5. Unde CHIAR ajută traiectoriile elicoidale / Hebbian / Lyapunov

Nu le arunca — repoziționează-le acolo unde arhitectura ta are avantaj real:
**disambiguare și încredere**, nu generarea răspunsului final.

### 5.1 Disambiguare de rol pentru cuvinte ambigue/noi
Când un cuvânt nu e în `lexicon.py` (ex. o formulare nouă, un sinonim, o greșeală
de scriere), **atunci** intervine partea învățată: bitboard-ul lexical fingerprint
al cuvântului necunoscut e comparat (Hamming distance / resonance) cu prototipurile
de rol învățate din antrenament (media bitboard-urilor tuturor cuvintelor cunoscute
cu ROLE=NUMBER, de exemplu). Dacă distanța e sub un prag → clasifică rolul prin
similaritate; altfel → UNKNOWN.

Aici traiectoria elicoidală chiar capătă sens: `Rij(t)` (raza care se restrânge spre
target) devine literal procesul de **convergență a scorului de încredere** pe măsură
ce se acumulează evidență (context) — nu un artificiu estetic peste rezultat.

### 5.2 Masa Hebbian → prag de încredere, nu scor de câștigător
`mass` (frecvența de activare a unui token) ar trebui folosită ca prag de
încredere pentru decizia de abstenție (§6), nu ca termen aditiv în alegerea
targetului (cum e acum, linia `+ target_state.mass * 0.1`).

---

## 6. Mecanism de abstenție (lipsește complet în v1)

Regulă explicită: dacă orice slot esențial (hour) rămâne `None` după parsare
**și** disambiguarea prin similaritate nu trece pragul de încredere →
`predict()` returnează `UNKNOWN`, nu un răspuns ghicit.

Test direct din conversația anterioară: `"banana o'clock"` → trebuie `UNKNOWN`,
nu `12:00`. Ăsta e testul de acceptanță #1 pentru v2.

---

## 7. Ce se schimbă per fișier din repo

| Fișier | Acțiune |
|---|---|
| `helibit/bitboard.py` | Adaugă `from_role_value`, `role()`, `numeric_value()` — păstrează API-ul vechi pentru compatibilitate |
| `helibit/lexicon.py` | **NOU** — dicționarul explicit de roluri (§3) |
| `helibit/parser.py` | **NOU** — extrage `slots` dintr-o secvență de tokeni/bitboard-uri via `role()` + `numeric_value()` |
| `helibit/arithmetic.py` | **NOU** — `resolve_time(slots)` (§4) |
| `helibit/engine.py` | `predict()` refactorizat: tokenize → parser.extract_slots → (disambiguare via Hebbian dacă slot lipsă) → arithmetic.resolve_time → UNKNOWN dacă None |
| `helibit/trainer.py` | Nu mai antrenează pe perechi (text, target_string) ca clasificare — antrenează prototipurile de rol (media bitboard pe ROLE) și pragurile de încredere pentru disambiguare |
| `helibit/dynamics.py` | Păstrat, dar `mass`/`velocity` migrează spre rol de "încredere", nu de "scor de câștig" |
| `dataset.py` / `dataset.json` | Reformulat: nu mai e (text, target_string) de memorat, ci (text, slots_așteptate) pentru evaluarea parserului + un test set compozițional separat (§8) |

---

## 8. Plan de testare riguros (obligatoriu, nu opțional)

1. **Split compozițional, nu random**: antrenament pe un subset de combinații
   (ore, direcții, modificatori), validare STRICT pe combinații absente din
   antrenament (exact ce am făcut manual în conversație — automatizează asta).
2. **Set adversarial**: input fără sens (`"banana o'clock"`), ordine greșită
   (`"past four quarter"`), format numeric (`"4:07"`) — toate trebuie tratate
   explicit (UNKNOWN sau pass-through), nu ghicite tăcut.
3. **Set de generalizare numerică**: minute care nu apar deloc în training
   (`"ten past four"`, `"twenty-five past three"`) — testează motorul aritmetic,
   nu memoria.
4. **Raport obligatoriu**: acuratețe separată pe (a) exact-match din training,
   (b) compoziții noi valide, (c) input adversarial/invalid — cele trei numere,
   nu un singur "100% acuratețe" agregat, care ascunde exact problema găsită.

---

## 9. Ce rămâne valid din propunerea inițială, ce trebuie reformulat

**Rămâne valid**: reprezentarea pe biți pentru viteză (AND/XOR/POPCNT), rularea
integer-only fără FPU, ideea de bitboard ca structură de date rapidă.

**Trebuie reformulat în document**: afirmațiile despre "formalizarea spațiului
semantic ca varietate Riemanniană" și "stabilitate Lyapunov dovedită" — în v2,
Lyapunov/traiectoria devin un mecanism de *încredere/disambiguare*, util și
onest, dar mult mai modest decât "motor de raționament semantic". Pentru orice
propunere de finanțare viitoare, descrie exact ce face v2: un parser de gramatică
formală + motor aritmetic + strat de disambiguare prin similaritate — nu un
înlocuitor general pentru Transformeri.

---

## 10. Prompt sugerat pentru Claude Code (rezumat executabil)

```
Refactorizează HeliBit-AI conform HELIBIT_ARCHITECTURE_V2.md:
1. Adaugă lexicon.py cu dicționarul de roluri explicit (§3)
2. Extinde Bitboard64 cu from_role_value/role()/numeric_value() (§2)
3. Creează parser.py: text -> slots (hour, minute_offset, direction, meridiem)
4. Creează arithmetic.py: slots -> "HH:MM" sau None (§4, cod dat ca referință)
5. Refactorizează engine.predict(): elimină candidate_targets/known_targets
   ca sursă de adevăr; folosește parser + arithmetic; adaugă UNKNOWN pentru
   sloturi nerezolvate
6. Repoziționează dynamics.py (mass/velocity) ca scor de încredere pentru
   disambiguarea tokenilor necunoscuți, nu ca termen de scor în alegerea
   targetului
7. Rescrie dataset.py cu split compozițional (train pe subset de combinații,
   test STRICT pe combinații absente) + set adversarial (§8)
8. Adaugă test_generalization.py care raportează separat: (a) exact-match,
   (b) compoziții noi valide, (c) input invalid/adversarial
```
