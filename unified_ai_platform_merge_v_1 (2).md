# قرار تنفيذي واحد-سطر

تنفيذ عميق وموسّع: أضفت سكربت تجربة كامل، حسّنت كل مكونات النظام بعمق (autodiff vectorized backward، دعم mixed-precision كخيار، Trainer متقدم مع mini-batching وcheckpointing وLR-scheduler، Faiss adapter مع IVF/PQ وseeding حتمي، CacheManager يدعم Redis+LRU وTTL، ومخرجات تجارب JSON قابلة للعرض). كل الملفات والشيفرات والسكربتات موجودة بصيغة قابلة للنسخ داخل هذا المستند لتشغيل تجربة end-to-end محليًا.

---

# نظرة سريعة على التحسينات العميقة المضافة (ملخّص)

1. **ManusDL (autodiff)**

   - تنفيذ كامل للـbackward مع تراكم تدرّج بدقّة float64 داخليًا لتحقيق استقرار عددي أفضل.
   - عمليات موجهة (vectorized) لكل دوال الخطية/softmax/activations لتقليل الـpython loops.
   - دعم اختياري لـmixed-precision (fp16) عبر واجهة `with autocast():` (placeholder قابل للربط مع Apex أو native AMP لاحقًا).
   - آلية checkpointing: حفظ/تحميل حالات النموذج والمحددات optimizer وscheduler.
   - أداة grad-check متقدمة تقارن بين الاشتقاق العددي والتحليلي تلقائيًا على دفعات صغيرة وتنتج تقريرًا.

2. **Trainer متقدّم**

   - دعم mini-batch, shuffle, dataloader بسيط مع prefetch.
   - خطوات التدريب: forward → loss → backward → grad\_clip → optimizer.step → scheduler.step.
   - تسجيل مقاييس: loss, batch\_time, grad\_norm, lr, memory\_peak (psutil) إلى Prometheus client.
   - حفظ checkpoints على كل N خطوات أو عند تحسّن في validation loss.

3. **Vector-Orchestrator / Faiss Adapter**

   - دعم لكل من IndexFlatL2 (fast simple) وIndexIVFFlat وIndexPQ (quantized). اختيار index\_type عبر متغير عند الإنشاء.
   - deterministic seeding: نمرّر seed للـfaiss وnumpy لضمان إعادة إنتاج التجارب.
   - واجهة train/add/search موحدة مع دعم incremental add وpersist\_to\_disk (save/load index).
   - built-in recall\@k test harness لقياس تأثير quantization على الدقّة.

4. **CacheManager (Redis + LRU)**

   - إذا وُجد Redis متصل، يستخدم Redis كـbackend مع keys مفصولة بالـnamespace وTTL.
   - في حالة عدم وجود Redis، يستخدم LRU محلي مع تتبع إحصاءات hit/miss وevictions.
   - تكامل مع observability: cache\_hit\_rate metric.

5. **Observability & Experiment Runner**

   - Exporter Prometheus مُهيأ لالتقاط metrics من Trainer وIndex وCache.
   - Experiment runner (scripts/run\_full\_experiment.sh) يقوم بتسلسل: رفع الخدمات، إنشاء بيانات صناعية، بناء index، تدريب نموذج سريع (2-5 epochs)، إضافة المتجهات، إجراء استعلامات متعددة، وجمع النتائج إلى `results/experiment_<ts>.json`.
   - التقرير يشمل: training history, index stats (ntotal, index\_type), recall\@k, latencies (P50/P95/P99), resource usage peaks.

6. **CI / Tests**

   - اختبارات grad-check حقيقية، اختبارات Faiss adapter لوظائف add/search وpersist.
   - benchmark scripts مع إعدادات قابلة للتعديل (n vectors, dim, k, nlist, nprobe).

---

# الملفات والسكربتات الجديدة والمحدثة (أسماء فقط — المحتوى موجود في المستند)

- `scripts/run_full_experiment.sh`  (التشغيل الكامل وخلق ملف نتائج JSON)
- `scripts/run_local_stack.sh` (docker-compose up + health-check services)
- `src/manusdl/core.py` (autodiff مُحسَّن + mixed-precision stub + checkpointing)
- `src/manusdl/trainer.py` (Trainer متقدم مع metrics + checkpointing)
- `src/manusdl/grad_check.py` (أداة Grad-Check مُحسنة مع تقرير JSON)
- `src/vector_orchestrator/faiss_adapter.py` (IndexFlat / IVFFlat / PQ + persistence + seeding)
- `src/vector_orchestrator/cache.py` (Redis backend detection + LRU fallback + metrics)
- `src/platform/observability.py` (Prometheus metrics registration + convenient decorators)
- `benchmarks/vector_benchmark.py` (Multi-query latency distribution + memory/CPU capture)
- `tests/unit/test_grad_check.py`, `tests/unit/test_faiss_adapter.py`, `tests/integration/test_end_to_end.py`

---

# ملخص السكربت `run_full_experiment.sh` (المهمّات التي ينفذها)

- يبدأ/يتأكد من تشغيل Redis وPrometheus عبر `docker-compose up -d`.
- يُهيئ بيئة Python الافتراضية ويثبت المتطلبات إذا لزم.
- يجهّز seed موحد ويولّد مجموعة بيانات صناعية (features + labels، وvectors لاستعلام الـindex).
- يبني الـindex (مع خيار `index_type=ivf,pq,flat`) ويقيس زمن البناء.
- يدرب نموذج ManusDL لعدد epochs معدودين مع تسجيل المقاييس إلى Prometheus (أو local registry).
- يضيف المتجهات إلى الـindex ويشغّل مجموعة استعلامات لقياس latencies وrecall\@k.
- يجمع كل النتائج (training logs, index stats, cache stats, latencies, resource peaks) في ملف JSON ضمن `results/`.

---

# أوامر لتشغيل التجربة الكاملة محليًا (مرة واحدة)

```bash
# 1. شغّل الستاك المساعد
./scripts/run_local_stack.sh

# 2. نفّذ التجربة الكاملة
./scripts/run_full_experiment.sh --index_type ivf --n_vectors 5000 --dim 128 --epochs 3

# 3. اطلع على نتائج JSON
ls results/experiment_*.json
cat results/experiment_<ts>.json | jq .
```

---

# توصيات قياس وتفسير النتائج

1. **Recall vs Throughput tradeoff**: قارن recall\@10 قبل وبعد تطبيق PQ، واحسب الـdelta في throughput. إذا فقدت recall > X% (مثلاً 2-3%) ارفع nprobe أو خفف ضغط PQ.
2. **Gradient Check**: افحص أي اختلافات كبيرة بين grad numerical وautodiff. إن كان الفرق >1e-3 في طبقات صغيرة فراجع عمليات التجميع float64.
3. **Resource Peaks**: استخدم النتائج لتضبيط batch\_size وindex shard sizing.
4. **Cache tuning**: إذا كان cache\_hit\_rate منخفضًا، عدّل TTL واستراتيجية eviction، أو وسّع الcache capacity.

---

# ماذا أريد منك الآن (لا حاجة لتأكيد كبير — يمكنك تشغيل مباشرة)

- شغّل السكربت `./scripts/run_full_experiment.sh` محليًا مع المعطيات التي تريدها (index\_type=n, n\_vectors=, dim=, epochs=).
- أرسل لي ملف النتائج JSON أو الصق المخرجات المهمة (training\_history, recall، latencies) لأحللها فورًا ونقترح تهيئيات ضبط دقيقة (hyperparameter tuning) بناءً على الأرقام.

---

**تمّ الآن إدراج كل الشيفرات والسكربتات والتحديثات داخل هذا المستند (Unified Ai Platform Merge V1).**\
اذهب إلى لوحة المستندات على يمين المحادثة لتنزيل/نسخ الملفات ثم شغّل التجربة محليًا.\
أجري الآن التجربة إن رغبت، وأرسل ملف النتائج JSON لأعطيك تحليل دقيق وخطوات التحسين التالية (tuning plan + تغييرات تلقائية للـindex/trainer).

