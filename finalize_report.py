"""Independent audit and report generation from LR1 artifacts; never sends model requests."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import shutil
from collections import Counter, defaultdict
from xml.sax.saxutils import escape


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def near(a, b):
    assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-8), (a, b)


def p90(values):
    x = sorted(values)
    n = .9 * (len(x)-1)
    i, j = math.floor(n), math.ceil(n)
    return x[i] + (x[j]-x[i])*(n-i)


def audit(source):
    root = Path(source)
    run = root / "runs/lr1_final"
    rows = lines(run / "results.jsonl")
    plan = lines(run / "requests.jsonl")
    metadata = read(run / "metadata.json")
    manifest = read(root / "data/manifest.json")["datasets"]["pride_and_prejudice"]
    checks = read(root / "data/checks.json")
    config = read(run / "config.yaml")
    passport = read(root / "server_passport.json")
    stored = read(root / "reports/summary.json")
    assert metadata["status"] == "completed"
    assert len(rows) == len(plan) == 320
    assert len({r["id"] for r in rows}) == len(rows)
    assert {r["id"] for r in rows} == {r["id"] for r in plan}
    assert all(r["status"] == "completed" for r in rows)
    assert digest(root / manifest["source_file"]) == manifest["source_sha256"]
    assert digest(root / manifest["prompts_file"]) == manifest["prompts_sha256"]
    assert digest(root / "data/warmups.jsonl") == manifest["warmups_sha256"]
    assert digest(root / "server_passport.json") == metadata["fingerprint"]["passport_sha256"]
    assert digest(root / "data/manifest.json") == metadata["fingerprint"]["manifest_sha256"]
    assert config["prompt_lengths"] == [256,1024,4096,8192,16384]
    assert config["output_lengths"] == [16,64,256,1024]
    assert config["concurrency"] == [1,2,4,8,16,32]
    if (root / "configs/experiment.json").exists():
        assert digest(root / "configs/experiment.json") == metadata["fingerprint"]["config_sha256"]
    groups = defaultdict(list)
    plan_by_id = {r["id"]:r for r in plan}
    for r in rows:
        p = plan_by_id[r["id"]]
        assert r["input_target"] == r["prompt_tokens"] == r["usage"]["prompt_tokens"] == p["input_target"]
        assert r["output_target"] == r["output_tokens"] == r["usage"]["completion_tokens"] == p["output_target"]
        assert r["finish_reason"] == "length", r["finish_reason"]
        assert r["t0"] <= r["t1"] <= r["tn"] <= r["request_end"]
        assert r["t1"] == r["chunks"][0]["time"]
        assert r["tn"] == r["chunks"][-1]["time"]
        assert "".join(c["text"] for c in r["chunks"]) == r["text"]
        near(r["ttft_s"], r["t1"]-r["t0"])
        near(r["tpot_s"], (r["tn"]-r["t1"])/(r["output_tokens"]-1))
        near(r["e2e_s"], r["tn"]-r["t0"])
        groups[r["group"]].append(r)
    aggregate = {}
    for g, rs in groups.items():
        assert len({r["segment"] for r in rs}) == 1
        span = max(r["tn"] for r in rs)-min(r["t0"] for r in rs)
        aggregate[g] = {
            "group":g, "n":len(rs), "input":rs[0]["input_target"], "output":rs[0]["output_target"],
            "c":rs[0]["concurrency"],
            "ttft":st.median(r["ttft_s"] for r in rs),
            "ttft_min":min(r["ttft_s"] for r in rs),
            "ttft_max":max(r["ttft_s"] for r in rs),
            "p90":p90([r["ttft_s"] for r in rs]),
            "tpot":st.median(r["tpot_s"] for r in rs),
            "e2e":st.median(r["e2e_s"] for r in rs),
            "tps":sum(r["output_tokens"] for r in rs)/span,
            "rps":len(rs)/span,
            "good":sum(r["ttft_s"] <= config["ttft_limit_s"] and r["tpot_s"] <= config["tpot_limit_s"] for r in rs),
        }
        a = aggregate[g]
        a["goodput"] = a["good"]/span
        a["prefill_tps"] = a["input"]/a["ttft"]
        a["passes"] = a["p90"] <= config["ttft_limit_s"] and a["tpot"] <= config["tpot_limit_s"]
    for s in stored["conditions"]:
        a = aggregate[s["group"]]
        assert a["n"] == s["planned"] == s["completed"]
        for k, v in [("ttft", "ttft_median_s"), ("p90", "ttft_p90_s"), ("tpot", "tpot_median_s"),
                     ("e2e", "e2e_median_s"), ("tps", "output_tokens_per_s"), ("rps", "completed_requests_per_s"),
                     ("goodput", "goodput_requests_per_s")]:
            near(a[k], s[v])
        a["waiting"] = s["max_waiting"]
        a["kv"] = s["max_kv_usage_fraction"]
    for name in ("prefill.csv", "decode.csv", "load.csv", "conditions.csv"):
        with (root / "reports" / name).open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                a = aggregate[row["group"]]
                near(float(row["ttft_median_s"]), a["ttft"])
                near(float(row["tpot_median_s"]), a["tpot"])
                assert int(row["planned"]) == a["n"]
    cache_plan = [p for p in plan if p["group"]=="cache"]
    assert all(p["messages"] == cache_plan[0]["messages"] for p in cache_plan)
    cache = sorted(groups["cache"],key=lambda r:r["id"])
    speedups = [cache[0]["ttft_s"]/r["ttft_s"] for r in cache[1:]]
    for x,y in zip(speedups,stored["prefix_cache_speedup_repeat2_repeat3"]):
        near(x,y)
    warmups = lines(run / "warmup.jsonl")
    assert len(warmups)==16 and all(r["status"]=="completed" for r in warmups)
    assert not {r["id"] for r in warmups} & {r["id"] for r in rows}
    assert checks["output_tokens"] == [64,64] and checks["texts_match"] is True
    # Verify a recorded example independently.
    example = rows[0]
    result = {
        "passed":True, "measurements":len(rows), "failed":0, "warmups_excluded":len(warmups),
        "groups":len(groups), "commit":metadata["fingerprint"]["commit"],
        "checked":"timestamps, N-1, usage lengths, CSV/summary agreement, all input hashes, warmups, cache prompts",
        "example":{"id":example["id"],"t0":example["t0"],"t1":example["t1"],"tn":example["tn"],
                   "N":example["output_tokens"],"ttft_s":example["ttft_s"],"tpot_s":example["tpot_s"]},
    }
    return aggregate, stored, passport, metadata, manifest, checks, result, cache


def report_content(source):
    a, summary, pas, meta, manifest, checks, audit_result, cache = audit(source)
    pre = [a[f"exp1_{n}"] for n in (256,1024,4096,8192,16384)]
    dec = [a[f"exp2_{n}"] for n in (16,64,256,1024)]
    load = [a[f"load_{n}"] for n in (1,2,4,8,16,32)]
    fit = summary["decode_fit"]
    bounds = summary["prefix_cache_speedup_repeat2_repeat3"]
    passed = max((x["c"] for x in load if x["passes"]), default=0)
    approx = pas["weight_file_bytes"]/300e9
    pages = []
    pages.append({
        "title":"ЛР1. Инференс и развёртывание языковой модели",
        "subtitle":"Стенд, прогноз и воспроизводимость",
        "paragraphs":[
            "<b>Вопрос.</b> Из чего складывается время ответа языковой модели и как оно меняется с длиной текста и числом одновременных пользователей?",
            "<b>Прогноз до опытов.</b> TTFT растёт с длиной входа, возможно быстрее линейного; повторный префикс снижает TTFT. Полное время примерно линейно зависит от длины ответа. Ожидаемый TPOT 30–80 мс, нижняя оценка около 27 мс. Предполагалось выполнение пределов при 1–4 клиентах и нарушение при 16–32; прогноз сохранён в configs/prediction.md.",
            f"<b>Стенд.</b> {pas['model']}, FP16; ревизия {pas['revision']}. vLLM {pas['server_version']}, PyTorch {meta['fingerprint']['versions']['torch']}, Python {meta['fingerprint']['python']}, Tesla T4, 15360 MiB памяти; 2 логических CPU. Число потоков отдельно не задавалось (переменные OMP/MKL/OpenBLAS не установлены).",
            "<b>Запуск.</b> Контекст 32768; gpu-memory-utilization=0.90; max-num-seqs=32; max-num-batched-tokens=2048; enforce-eager; prefix caching включён; auto tool choice с parser=hermes. Полная команда, версия драйвера и журнал сохранены в паспорте и data/server_startup.log. Исходный запуск не закреплял --revision; ревизия восстановлена из локального HF snapshot. Для повторения следует явно указать её.",
            "<b>Данные и протокол.</b> Pride and Prejudice (Project Gutenberg №1342). Длины входов задаются токенизатором модели после шаблона диалога и проверяются по usage.prompt_tokens; длины ответов — по usage.completion_tokens. Temperature=0, остальные параметры явные, ignore_eos=True. Метка стоит первой; уникальный cache_salt изолирует обычные запросы. В опыте кеша текст и соль одинаковы.",
            f"<b>Память.</b> Веса: {pas['weight_file_bytes']/1e9:.3f} GB ({pas['weight_file_bytes']/2**30:.3f} GiB). KV на токен: 2 × 36 × 8 × 128 × 2 = 147456 байт = 144 KiB. Кеш 32768 токенов — 4.50 GiB. Выделено {pas['kv_capacity_tokens']} токенов, то есть {pas['kv_memory_calculated_gib']:.3f} GiB; журнал сообщает {pas['kv_memory_log_gib']} GiB. Для 8 пользователей по 4000 токенов нужно {summary['kv_8_users_4000_tokens_gib']:.3f} GiB: по ёмкости помещается, но скорость этим не гарантируется.",
            f"<b>Проверки.</b> Ровно 64 токена в обоих проверочных ответах; тексты при temperature=0 совпали. Пример t0=0, t1=0.4, tN=2.4, N=101 даёт TTFT=400 мс и TPOT=20 мс. Сохранены все 320 измерений без ошибок; 16 прогревов исключены. Результаты и времена фрагментов — в results.jsonl, запуски — в MLflow.",
            "<b>Воспроизводимость.</b> Код проверял чистый Git, сохранял итоговую конфигурацию, версии, паспорт и SHA-256. Коммит измерений: "+meta["fingerprint"]["commit"]+". История доступна в source_history.bundle архива артефактов. Числа ниже независимо пересчитаны из первичных записей.",
        ],
        "table":None,"image":None,
        "footnote":"SHA-256 исходного текста: "+manifest["source_sha256"],
    })
    pages.append({
        "title":"Эксперимент 1. Длина запроса и TTFT",
        "subtitle":"Серия exp1_*: 5 запросов на длину, ответ 16 токенов",
        "paragraphs":[
            "Запросы выполнялись по одному. Перед каждой группой один отдельный прогрев. Между измерительными запросами повторное использование префикса исключено меткой и cache_salt; сервер при этом сохранял один профиль. TTFT включает клиентскую отправку, очередь и обработку входа.",
        ],
        "table":[["Вход,\nтокены","Успехи /\nплан","TTFT,\nмедиана, с","Диапазон,\nс","Входные\nтокены/с"]]+[
            [str(x["input"]),"5 / 5",f'{x["ttft"]:.4f}',f'{x["ttft_min"]:.4f}–{x["ttft_max"]:.4f}',f'{x["prefill_tps"]:.1f}'] for x in pre],
        "image":"prefill.png",
        "after":[
            f"TTFT вырос с {pre[0]['ttft']:.3f} до {pre[-1]['ttft']:.3f} с. При удвоениях 4096→8192 и 8192→16384 рост составил {pre[3]['ttft']/pre[2]['ttft']:.2f} и {pre[4]['ttft']/pre[3]['ttft']:.2f} раза. Рост быстрее линейного; скорость вход/TTFT снизилась с {pre[0]['prefill_tps']:.0f} до {pre[-1]['prefill_tps']:.0f} токенов/с. Это согласуется с ростом затрат внимания при длинном контексте. Влияние конкретного backend и eager-режима отдельно не изолировано.",
            f"<b>Prefix caching.</b> Для трёх одинаковых запросов по 8192 токена TTFT: {cache[0]['ttft_s']:.4f}; {cache[1]['ttft_s']:.4f}; {cache[2]['ttft_s']:.4f} с. Повторы ускорились в {bounds[0]:.1f} и {bounds[1]:.1f} раза. Экономится повторный prefill, а не генерация ответа. Агент может переиспользовать общий префикс с описаниями 24 инструментов и началом истории.",
        ],
        "footnote":"Источник: runs/lr1_final/results.jsonl; таблица reports/prefill.csv. Скорость вход/TTFT — клиентская оценка, не чистая производительность GPU prefill.",
    })
    pages.append({
        "title":"Эксперимент 2. Длина ответа и decode",
        "subtitle":"Серия exp2_*: вход 1024 токена, по 5 запросов на длину ответа",
        "paragraphs":[
            "Длина ответа принудительно фиксировалась ignore_eos и проверялась по usage. TPOT=(tN−t1)/(N−1): первый токен уже учтён в TTFT. Метрики считаются по числу токенов сервера, а не числу потоковых фрагментов.",
        ],
        "table":[["Выход,\nтокены","Успехи /\nплан","Полное время,\nмедиана, с","TPOT,\nмедиана, мс","TTFT,\nмедиана, с"]]+[
            [str(x["output"]),"5 / 5",f'{x["e2e"]:.4f}',f'{x["tpot"]*1000:.2f}',f'{x["ttft"]:.4f}'] for x in dec],
        "image":"decode.png",
        "after":[
            f"Аппроксимация по четырём медианам: T(N)={fit['intercept_s']:.4f}+{fit['slope_s_per_token']:.5f}×(N−1) с. Наклон — {fit['slope_s_per_token']*1000:.2f} мс/токен, или около {1/fit['slope_s_per_token']:.1f} токена/с. Свободный член {fit['intercept_s']:.3f} с близок по масштабу к наблюдаемому TTFT {min(x['ttft'] for x in dec):.3f}–{max(x['ttft'] for x in dec):.3f} с, но не равен ему: это оценка прямой по четырём точкам, а TPOT не строго постоянен.",
            f"Измеренные медианы TPOT {min(x['tpot'] for x in dec)*1000:.1f}–{max(x['tpot'] for x in dec)*1000:.1f} мс меняются намного слабее, чем длина ответа. Нижняя оценка по весам и 300 GB/с составляет {approx*1000:.1f} мс. Наклон примерно в {fit['slope_s_per_token']/approx:.2f} раза выше: кроме чтения весов есть KV-кеш, вычисления, планировщик и клиентский streaming. Теоретический предел предполагает идеальное использование полосы памяти.",
            "Прогноз примерно линейного роста полного времени и TPOT в пределах 30–80 мс подтвердился. При длинном входе decode тоже может замедляться из-за большего кеша, поэтому результат нельзя переносить на произвольный контекст.",
        ],
        "footnote":"Источник: runs/lr1_final/results.jsonl; reports/decode.csv и summary.json. Интервал полного ответа заканчивается последним фрагментом текста, а не последующим пакетом usage.",
    })
    pages.append({
        "title":"Эксперимент 3. Нагрузка и пределы сервера",
        "subtitle":"Серия load_*: вход 1024, выход 128 токенов; max(16,4c) запросов",
        "paragraphs":[
            "Каждый из c клиентов отправлял следующий запрос после ответа. Пределы заданы до опыта: p90 TTFT ≤2 с и медиана TPOT ≤100 мс. Goodput — число запросов, выполнивших оба индивидуальных предела, за секунду уровня. Знаменатель времени — от первой отправки до последнего текста. Все уровни непрерывные, без возобновлений.",
        ],
        "table":[["Клиенты","Успехи /\nплан","p90 TTFT,\nс","Мед. TPOT,\nмс","Выход,\nтокены/с","Goodput,\nзапросы/с"]]+[
            [str(x["c"]),f'{x["n"]} / {x["n"]}',f'{x["p90"]:.3f}',f'{x["tpot"]*1000:.1f}',f'{x["tps"]:.2f}',f'{x["goodput"]:.3f}'] for x in load],
        "image":"throughput_latency.png",
        "after":[
            f"<b>Предел — {passed} клиента среди проверенных уровней.</b> При c=2 p90 TTFT={load[1]['p90']:.3f} с, медиана TPOT={load[1]['tpot']*1000:.1f} мс. Один запрос из 16 превысил индивидуальный TTFT: поэтому goodput {load[1]['goodput']:.3f} ниже общего {load[1]['rps']:.3f} запроса/с. При c=4 нарушается TTFT, а при c=16 — уже оба предела. Прогноз прохождения всего диапазона 1–4 подтвердился частично.",
            f"Пропускная способность выросла с {load[0]['tps']:.2f} до {load[-1]['tps']:.2f} токена/с ({load[-1]['tps']/load[0]['tps']:.2f} раза), но задержки ухудшились и goodput при c≥8 равен нулю. Батч позволяет совместно читать веса, однако отдельный запрос обслуживается медленнее.",
            "Наблюдавшиеся максимумы ожидающих: "+", ".join(f'{x["waiting"]:.0f}' for x in load)
            +"; KV-кеша: "+", ".join(f'{100*x["kv"]:.1f}%' for x in load)+". "
            +f"Очередь появилась при c={next(x['c'] for x in load if x['waiting']>0)}, "
            +"когда кеш далёк от заполнения: раннюю задержку нельзя приписать нехватке KV. "
            +f"При c=32 кеш близок к ёмкости. Верхняя оценка памяти {pas['kv_capacity_tokens']}/(1024+128)"
            +f"≈{summary['kv_capacity_short_requests_upper_bound']} запроса; max-num-seqs=32 и задержки ограничивают полезную нагрузку раньше.",
            "<b>Границы вывода.</b> Максимумы сняты с шагом около секунды, короткие пики могли быть пропущены. Вывод относится к этому T4, eager-режиму и профилю запросов. Для ЛР2–ЛР8 разумная исходная параллельность — 1–2; увеличение документов и описаний инструментов требует новых замеров. Результаты не обобщаются на оптимизированный vLLM и другое оборудование.",
            "<b>Кодогенерация.</b> Код подготовлен с ИИ; проверен 12 тестами, известным примером времени, длинами usage, повтором при temperature=0, хешами и независимым пересчётом 320 записей. Таблицы и графики созданы скриптом; неуспехов нет, прогрев исключён.",
        ],
        "footnote":"Источники: задание dm780009/intsys/LR1_llm_inference_serving.pdf; Qwen/Qwen3-4B-Instruct-2507; docs.vllm.ai/en/v0.33.0; Project Gutenberg №1342. Полные ссылки — в README.",
    })
    def normalize(text):
        return text.replace("–","-").replace("—","-").replace("‑","-").replace("−","-")
    for page in pages:
        for key in ("title","subtitle","footnote"):
            page[key] = normalize(page[key])
        for key in ("paragraphs","after"):
            if key in page:
                page[key] = [normalize(t) for t in page[key]]
        if page["table"]:
            page["table"] = [[normalize(t) for t in row] for row in page["table"]]
    return pages, audit_result


def markdown(pages):
    result = []
    for i,page in enumerate(pages):
        result.append(("# " if i==0 else "## ")+page["title"])
        result.append(page["subtitle"])
        for p in page["paragraphs"]:
            result.append(p.replace("<b>","**").replace("</b>","**"))
        if page["table"]:
            table = page["table"]
            block = ["|"+"|".join(x.replace("\n"," ") for x in table[0])+"|",
                     "|"+"|".join("---" for _ in table[0])+"|"]
            block.extend("|"+"|".join(row)+"|" for row in table[1:])
            result.append("\n".join(block))
        if page["image"]:
            result.append("!["+page["title"]+"](figures/"+page["image"]+")")
        result.extend(p.replace("<b>","**").replace("</b>","**") for p in page.get("after",[]))
        result.append(page["footnote"])
        result.append("")
    return "\n\n".join(result)+"\n"


def pdf(pages, output, figures):
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Paragraph, Table, TableStyle
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.colors import HexColor, white
    from reportlab.lib.utils import ImageReader
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.enums import TA_LEFT
    import os
    font_root = Path(os.environ.get("WINDIR","C:/Windows")) / "Fonts"
    if (font_root / "arial.ttf").exists():
        regular, bold = font_root/"arial.ttf", font_root/"arialbd.ttf"
    else:
        regular = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
        bold = regular.with_name("DejaVuSans-Bold.ttf")
    pdfmetrics.registerFont(TTFont("Body",str(regular)))
    pdfmetrics.registerFont(TTFont("BodyBold",str(bold)))
    pdfmetrics.registerFontFamily("Body",normal="Body",bold="BodyBold")
    w,h = A4
    left, width = 39, w-78
    c = canvas.Canvas(str(output), pagesize=A4)
    c.setTitle("ЛР1. Инференс и развёртывание языковой модели")
    c.setAuthor("Лабораторная работа")
    for number,page in enumerate(pages,1):
        c.setFillColor(HexColor("#193b57"))
        c.rect(0,h-12,w,12,fill=1,stroke=0)
        c.setFont("BodyBold",15 if number==1 else 14)
        c.drawString(left,h-47,page["title"])
        c.setFont("Body",9)
        c.setFillColor(HexColor("#566575"))
        c.drawString(left,h-65,page["subtitle"])
        y = h-85
        font = 9.4 if number==1 else (8.6 if number==4 else 9.5)
        style = ParagraphStyle("body",fontName="Body",fontSize=font,leading=font*1.34,textColor=HexColor("#172b3a"),alignment=TA_LEFT)
        def para(text, small=False):
            nonlocal y
            s = ParagraphStyle("small",parent=style,fontSize=7.4,leading=9.5) if small else style
            p=Paragraph(text,s)
            _,ph=p.wrap(width,750)
            p.drawOn(c,left,y-ph)
            y -= ph+7
        for text in page["paragraphs"]:
            para(text)
        if page["table"]:
            table = [[Paragraph(escape(cell).replace("\n","<br/>"),
                      ParagraphStyle("cell",fontName="BodyBold" if j==0 else "Body",
                                     fontSize=7.9,leading=10,textColor=white if j==0 else HexColor("#172b3a")))
                      for cell in row] for j,row in enumerate(page["table"])]
            t=Table(table,colWidths=[width/len(table[0])]*len(table[0]))
            t.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,0),HexColor("#193b57")),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),[HexColor("#eff4f8"),white]),
                ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
                ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),5),
                ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
            ]))
            _,th=t.wrap(width,700)
            t.drawOn(c,left,y-th)
            y -= th+8
        if page["image"]:
            img=ImageReader(str(figures/page["image"]))
            iw,ih=img.getSize()
            image_height = width*ih/iw
            if number==4:
                image_height = min(image_height, 207)
            image_width = image_height*iw/ih
            c.drawImage(img,left+(width-image_width)/2,y-image_height,
                        width=image_width,height=image_height,mask="auto")
            y-=image_height+8
        for text in page.get("after",[]):
            para(text)
        para(page["footnote"],small=True)
        if y < 34:
            raise ValueError(f"PDF page {number} overflow: y={y:.1f}")
        c.setFont("Body",8)
        c.setFillColor(HexColor("#566575"))
        c.drawString(left,23,"Интеллектуальные системы · ЛР1 · Серия lr1_final")
        c.drawRightString(w-left,23,f"{number} / 4")
        c.showPage()
    c.save()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",default=".")
    ap.add_argument("--output",default="reports")
    ap.add_argument("--pdf",action="store_true")
    args=ap.parse_args()
    source,output=Path(args.source),Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    pages,result=report_content(source)
    (output/"report.md").write_text(markdown(pages),encoding="utf-8")
    (output/"audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    figures=output/"figures"
    figures.mkdir(exist_ok=True)
    for p in (source/"reports/figures").glob("*.png"):
        dest=figures/p.name
        if p.resolve()!=dest.resolve():
            shutil.copyfile(p,dest)
    if args.pdf:
        pdf(pages,output/"report.pdf",figures)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    print("Report generated from audited primary records:",output.resolve())


if __name__=="__main__":
    main()
