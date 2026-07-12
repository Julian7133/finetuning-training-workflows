# Implementation Plan v4: Fine-Tuning Run (gestuft: SFT → bedingt DPO)

Follow-up zu Plan v2 (Completion-Aware Curation) und v3 (Self-Replay). Die Datenpipeline ist fertig (Stage 3+5 laut Rückmeldung sicher, Stage-4-DPO-Pairs Status unklar — Schritt 0 klärt das zuerst). Dieser Plan deckt den eigentlichen Trainingslauf ab: Option 3 aus der Diskussion (gestuft, SFT zuerst, DPO nur bei Bedarf).

## Schritt 0 — Bestandsaufnahme (zuerst, vor allem anderen)

- Prüfen, ob `build_dpo.py` (Plan v2, Stage 4) je gelaufen ist und eine Output-Datei existiert. Falls ja: Größe/Format notieren, aber **noch nicht verwenden** — kommt erst in Phase 2 zum Einsatz, falls überhaupt nötig.
- Finale Größe des gemischten SFT+Replay-Trainingssets (Stage 5d-Output) exakt bestimmen — laut Einschätzung 500-2000 Beispiele, für die konkrete Hyperparameter-Wahl unten aber die reale Zahl nutzen.
- Sequenzlängen-Verteilung im Trainingsset prüfen (nicht nur Anzahl Beispiele): OpenClaw-Tool-Call-Trajektorien mit Tool-Ergebnissen können deutlich länger sein als die üblichen 512-Token-Beispiele aus Standard-LoRA-Guides — das bestimmt `max_seq_length` und damit den tatsächlichen Speicherbedarf beim Training, nicht die Beispielanzahl allein.

## Schritt 0.5 — Modus-Auswahl (interaktiv, zu Beginn)

Standardmäßig läuft **Option 1** (reines SFT, kein Eval-Gate, direkter Export nach Phase 1 — kein DPO). Vor dem eigentlichen Trainingsstart einmalig interaktiv fragen:

```
Standard-Lauf (SFT only) [Enter/1] oder gestufter Lauf mit DPO-Evaluation (3)? >
```

- Keine Eingabe / Enter / "1" → **Option 1**: nach Phase 1 direkt zu Phase 3 (Export). Phase 1.5 und Phase 2 werden komplett übersprungen.
- "3" → **Option 3** (wie im Rest dieses Plans beschrieben): Phase 1.5 (Evaluation) läuft nach dem SFT-Training, Phase 2 (DPO) folgt bedingt auf deren Ergebnis.

**Sonderfall, aktuell relevant:** Es wurden bereits geprüft — es existieren noch keine DPO-Pairs (`build_dpo.py` aus Plan v2 ist noch nicht gelaufen), und wie lange die Erzeugung dauern würde, ist unbekannt. Falls bei der Modus-Abfrage "3" gewählt wird, aber Schritt 0 bestätigt, dass keine DPO-Pairs vorliegen: **nicht automatisch `build_dpo.py` starten.** Stattdessen eine klare Warnung ausgeben, dass dieser Zwischenschritt zuerst nötig wäre und die Laufzeit dafür noch nicht abschätzbar ist, und explizit nachfragen, ob (a) `build_dpo.py` jetzt gestartet werden soll (mit dem Risiko unbekannter Dauer), oder (b) vorerst doch mit Option 1 fortgefahren werden soll, während `build_dpo.py` parallel/separat läuft.



### Modell & Methode
- Basis: `andjiang/CoPaw-Flash-9B-oQ4` (bereits verifiziert, siehe Plan v3) — QLoRA-Stil, LoRA-Adapter auf dem 4-bit-Modell. Bei 9B auf 24GB Unified Memory komfortabel machbar (LoRA friert die Base-Weights ein, Optimizer-State skaliert nur mit den Adapter-Parametern, nicht mit dem ganzen Modell).
- Tool: `mlx_lm.lora --train`, Standardweg für LoRA-Training auf Apple Silicon.

### Hyperparameter (Startwerte, 2026-Konsens für diese Größenordnung)
- Rank r=16, Alpha=16 (1:1-Skalierung — stabiler, gut erprobter Default; nicht das aggressivere α=2r, das mehr Tuning-Aufwand bräuchte).
- `target_modules`: alle linearen Layer ("all-linear"), nicht nur Attention — 2026-Standardempfehlung, bessere Ergebnisse bei überschaubarem Mehraufwand.
- Learning Rate: 1e-4 (sicherer Startwert im empfohlenen 1e-4–2e-4-Band).
- Epochen: 2 als Startwert bei 500-2000 Beispielen — bei so kleinem Set ist Überanpassung das größere Risiko als Unteranpassung, lieber konservativ starten und bei Bedarf eine dritte Epoche nachlegen, als von Anfang an 3+ zu fahren.
- Gradient Checkpointing: an — reduziert Aktivierungsspeicher, wichtig bei potenziell langen Tool-Call-Sequenzen (siehe Schritt 0).
- Batch Size: 1 mit Gradient Accumulation (Ziel-effektive Batchgröße 8-16) — bei variabler, teils langer Sequenzlänge in Agent-Trajektorien ist das der robustere Startpunkt als eine feste größere Batchgröße.

### Eval-Split (wichtig, nicht Standard-Zufallssplit)
- Kein rein zufälliger Train/Dev-Split. Stattdessen gezielt einen kleinen Anteil (~10%) der Episoden mit `completion_status == completed_after_nudge` oder `abandoned_*` (aus der Judge-Stage, Plan v2) zurückhalten — das sind genau die Fälle, an denen sich zeigt, ob das Loop-Completeness-Problem durch SFT allein behoben wird.

## Phase 1.5 — Evaluation (Entscheidungspunkt, nur bei Option 3)

- Den fertigen LoRA-Adapter (ohne Merge, direkt über `mlx_lm.generate --adapter-path ...` — schnelles Testen ohne Fuse-Schritt) gegen die zurückgehaltenen Nudge-/Abandoned-Episoden laufen lassen.
- Manuell (oder mit Kimi-K2.7-Judge, gleiche Infrastruktur wie in Plan v3) prüfen: setzt das Modell nach einem Tool-Ergebnis jetzt autonom fort, oder stoppt es weiterhin und wartet?
- **Entscheidung:** Falls die Fortsetzungsrate klar besser ist als beim Base-Modell → weiter zu Phase 3 (Export, Plan v3 Stage 6), DPO-Stufe entfällt. Falls das Problem im Kern bestehen bleibt → weiter zu Phase 2.

## Phase 2 — DPO-Stufe (nur bei Option 3, und nur falls Phase 1.5 das nötig macht)

- Voraussetzung: Schritt 0.5 hat "3" gewählt UND Preference-Pairs aus Stage 4 existieren (aktuell: existieren noch nicht — `build_dpo.py` müsste vorher separat laufen, Laufzeit unbekannt).
- Offener technischer Punkt, vor Start zu klären: ob `mlx_lm.lora` DPO-Pairs direkt verarbeiten kann, oder ob TRL mit MPS-Backend nötig ist (langsamer, aber lauffähig) — kurzer Rechercheschritt, bevor Code geschrieben wird.
- Trainiert auf dem bereits SFT-angepassten Adapter weiter (nicht auf der rohen Base) — DPO als zweite, aufbauende Stufe, kein Neustart von vorne.

## Phase 3 — Export

- Wie in Plan v3, Stage 6 bereits vorbereitet: `mlx_lm.fuse` → `convert_hf_to_gguf.py` → `quantize` für den ThinkPad-Einsatz. Kein neuer Schritt, nur der bereits geplante angewendet auf den finalen Adapter (SFT-only oder SFT+DPO, je nach Phase-1.5-Ergebnis).

## Reihenfolge

1. Schritt 0 (Bestandsaufnahme) — wenige Minuten, klärt zwei offene Unbekannte.
2. Schritt 0.5 (Modus-Auswahl) — interaktive Abfrage, Default Option 1.
3. Phase 1 (SFT-Training) — bei 500-2000 Beispielen und Rank 16 auf einem M3 realistisch im Bereich von ein bis mehreren Stunden, abhängig von der tatsächlichen Sequenzlänge aus Schritt 0.
4. Bei Option 1: direkt weiter zu Phase 3 (Export).
5. Bei Option 3: Phase 1.5 (Evaluation) — Entscheidungspunkt, keine Umsetzung von Phase 2 ohne dieses Ergebnis.
6. Phase 2 (DPO) — bedingt, nur falls 1.5 es zeigt (Option 3) und DPO-Pairs vorliegen.
7. Phase 3 (Export) — wie geplant.

---

## Prompt für den Coding-Agenten (Claude Code / OpenClaw mit Repo-Zugriff)

```
Kontext: Die Datenpipeline (Plan v2 + v3) ist fertig. Jetzt folgt der
eigentliche Fine-Tuning-Lauf, gestuft: erst SFT, dann eine Evaluation, die
entscheidet, ob eine zusätzliche DPO-Stufe nötig ist.

Bitte in dieser Reihenfolge umsetzen, nach jedem Schritt kurz Zwischenstand
zeigen, bevor der nächste beginnt:

1. Bestandsaufnahme (kein Code-Schritt, nur Diagnose): Prüfen, ob eine
   Output-Datei von build_dpo.py (Plan v2, Stage 4) existiert. Falls ja,
   Zeilenzahl und Format kurz berichten. Aktueller bekannter Stand: es
   existieren noch KEINE DPO-Pairs. Die exakte Zeilenzahl des finalen
   SFT+Replay-Trainingssets (Stage 5d-Output) ausgeben. Die Token-Längen-
   Verteilung dieses Sets berechnen (min/median/p90/max), um eine sinnvolle
   max_seq_length für das Training abzuleiten — NICHT pauschal 512 annehmen,
   da Agent-Trajektorien mit Tool-Ergebnissen deutlich länger sein können.

2. Interaktive Modus-Abfrage (neu, vor dem eigentlichen Training): Beim
   Start des Trainings-Orchestrierungsskripts einmalig per CLI-Prompt fragen:
   "Standard-Lauf (SFT only) [Enter/1] oder gestufter Lauf mit
   DPO-Evaluation (3)?". Keine Eingabe oder "1" → Modus "sft_only" (Default).
   "3" → Modus "staged_dpo". Falls "staged_dpo" gewählt wird, aber Schritt 1
   bereits festgestellt hat, dass keine DPO-Pairs existieren: NICHT
   automatisch build_dpo.py starten. Stattdessen klar warnen, dass dieser
   Zwischenschritt zuerst nötig wäre und die Laufzeit unbekannt ist, und
   per weiterer CLI-Abfrage fragen, ob (a) build_dpo.py jetzt gestartet
   werden soll, oder (b) stattdessen doch mit Modus "sft_only" fortgefahren
   werden soll.

3. train_sft.py (neu, oder direkter mlx_lm.lora-Aufruf mit Config-Datei):
   LoRA-Training auf andjiang/CoPaw-Flash-9B-oQ4 mit dem finalen SFT+Replay-
   Set. Läuft identisch in beiden Modi. Hyperparameter: rank=16, alpha=16,
   target_modules=all-linear, learning_rate=1e-4, epochs=2 (als Startwert,
   nicht fix), gradient checkpointing an, batch_size=1 mit Gradient
   Accumulation Richtung effektiver Batchgröße 8-16, max_seq_length aus
   Schritt 1 ableiten statt raten. Vor dem Split: gezielt ~10% der Episoden
   mit completion_status completed_after_nudge oder abandoned_* (aus der
   Judge-Stage, Plan v2) als Eval-Set zurückhalten, nicht zufällig splitten
   — auch im Modus "sft_only" schon so zurückhalten, falls später doch
   evaluiert werden soll, auch wenn Phase 1.5 in diesem Modus nicht
   automatisch läuft.

4. Bei Modus "sft_only": nach Schritt 3 direkt zu Phase 3 (Export, bereits
   in Plan v3 Stage 6 vorbereitet) übergehen. Kein eval_completion.py-Lauf.

5. Bei Modus "staged_dpo": eval_completion.py (neu): Den trainierten
   Adapter unfusioniert (über mlx_lm.generate mit --adapter-path) gegen
   das in Schritt 3 zurückgehaltene Eval-Set laufen lassen. Für jede
   Episode prüfen, ob das Modell nach einem Tool-Ergebnis autonom fortsetzt
   oder stoppt/nach Bestätigung fragt — entweder über die bestehende
   Kimi-K2.7-Judge-Infrastruktur (leichte Rubrik: continued_autonomously:
   bool) oder als einfache Heuristik (Antwort endet mit einer Frage/
   Aufforderung vs. mit einem weiteren Tool-Call). Eine zusammenfassende
   Fortsetzungsrate ausgeben und mit dem unfeingetunten Base-Modell auf
   denselben Eval-Beispielen vergleichen.

6. Nach Schritt 5: kurz zusammenfassen, ob die Fortsetzungsrate sich klar
   verbessert hat. NICHT eigenständig entscheiden, ob Phase 2 (DPO) folgt —
   das Ergebnis berichten und auf Rückmeldung warten, bevor mit einer
   DPO-Implementierung begonnen wird, da diese Phase zusätzlichen
   Recherche- und Setup-Aufwand hat (mlx-lm-DPO-Support vs. TRL/MPS-
   Fallback muss erst geklärt werden).

Nicht in scope für diesen Task: die DPO-Stufe selbst (Phase 2) und der
finale GGUF-Export (Phase 3, bereits in Plan v3 vorbereitet) — beide erst
nach Rückmeldung zum Evaluationsergebnis.

Bitte an den bestehenden Code-Stil und die Fehlerbehandlung aus den
vorherigen Stages halten, keine neuen Abhängigkeiten ohne Rückfrage
einführen (mlx-lm ist bereits vorausgesetzt).
```
