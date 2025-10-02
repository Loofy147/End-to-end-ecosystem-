# خريطة هرمية علائقية لوكيل UnifiedAgent — نسخة 1

## نظرة عامة

هذه الوثيقة تصف خريطة أهداف فرعية هرمیّة ذات علاقات (Hierarchical Relational Subgoals Map) قابلة للتحويل مباشرة إلى بنية برمجية (Graph modules / Graph DB) لوكيلك الموحد (UnifiedAgent). الخريطة ترتكز على مكونات المشروع الموجودة: UnifiedPuzzleEnv, ProblemIdentifier, StrategySelector, UnifiedAgent, HierarchicalRLAgent, و linear algebra solver.

---

# مكونات الخريطة (Nodes)

1. **RootGoal**
   - وصف: وكيل متعدد الإستراتيجيات يتعرف على نوع اللغز (ملاحة / هندسة عكسية) ويحلّه بكفاءة.

2. **ProblemIdentifier**
   - دور: تمييز نوع اللغز (navigation vs reverse-engineering).
   - مخرجات: {type: "navigation"|"reverse_engineering"|"hybrid", confidence: float}

3. **StrategySelector**
   - دور: اختيار الإستراتيجية المناسبة أو توليف إستراتيجيات فرعية بناءً على مخرجات ProblemIdentifier.
   - مخرجات: strategy_spec (قائمة من وحدات التنفيذ وترتيبها وأوزانها).

4. **HierarchicalRLAgent**
   - دور: حل ألغاز الملاحة باستخدام Hierarchical RL (policies متداخلة: high-level subgoals + low-level controllers).
   - نقاط ضعف حالية: يحتاج تدريبًا مكثفًا على سيناريوهات متعددة.

5. **LinearAlgebraSolver**
   - دور: استخراج معاملات الدوال في ألغاز الهندسة العكسية (linear / least squares) وتوسيع لدعم non-linear.

6. **UnifiedAgent**
   - دور: كائن تكاملي ينفّذ تدفق العمل: استدعاء ProblemIdentifier → StrategySelector → تنفيذ الاستراتيجية → تسجيل النتيجة.

7. **TrainingDatasetRepo**
   - دور: مستودع سيناريوهات التدريب والاختبار (navigation maps, reverse-engineering examples, hybrid cases).

8. **CurriculumManager**
   - دور: توليد مسارات تدريب (من السهل إلى المعقّد) وإدارة sequencing و sampling.

9. **Logger & Evaluator**
   - دور: تجميع مقاييس الأداء، تسجيل الحوادث، توليد تقارير التجارب.

10. **ScenarioGenerator**
    - دور: إنشاء خرائط ملاحة ديناميكية، حالات أهداف متعددة، حالات عشوائية بصيغ مختلفة، وضجيج في بيانات الإدخال.

11. **ModelZoo**
    - دور: تخزين النسخ المدربة من HierarchicalRLAgent ونسخ محسنّة من solvers.

12. **InterfaceLayer (API/CLI)**
    - دور: طبقة تفاعل خارجية لتشغيل التجارب، استدعاء حلول فردية، أو تصدير نتائج.

---

# العلاقات (Edges)

- RootGoal → ProblemIdentifier (depends_on)
- ProblemIdentifier → StrategySelector (provides_input)
- StrategySelector → {HierarchicalRLAgent, LinearAlgebraSolver} (routes)
- TrainingDatasetRepo → {HierarchicalRLAgent, LinearAlgebraSolver, CurriculumManager} (supplies_data)
- CurriculumManager → HierarchicalRLAgent (trains)
- ScenarioGenerator → TrainingDatasetRepo (populates)
- UnifiedAgent → Logger & Evaluator (reports)
- ModelZoo → UnifiedAgent (loads_models)
- InterfaceLayer → UnifiedAgent (invoke)

---

# مواصفات العقد — واجهات وملفات التعريف (Node Schemas)

## ProblemIdentifier
- مدخل: observation (env data) أو I/O examples list
- مخرجات: {type, confidence, features}
- خوارزمية مقترحة: ensemble من: rule-based detectors + small classifier (ML) + pattern-matching
- اختبارات: precision/recall على مجموعة validation

## StrategySelector
- مدخل: {type, confidence, features}
- مخرجات: strategy_spec = {primary: module_name, fallback: [module_name], blend_weights: {}}
- سياسة عند غموض: استخدام وزن احتمالي وبدء hybrid run (جزئي لكل استراتيجية)

## HierarchicalRLAgent
- مدخل: env state
- مخرجات: action
- بنية مقترحة:
  - High-level policy: يحدد subgoal sequence
  - Low-level controllers: تتحكم بحركات خطوة بخطوة
- تقنيات تدريب: PPO/IMPALA+options، curriculum learning، intrinsic rewards (progress-to-subgoal)

## LinearAlgebraSolver
- مدخل: input_output_pairs
- مخرجات: solution (coefficients, fit_quality)
- خوارزميات: least squares (linreg), SVD, ridge regression, nonlinear least squares (Levenberg–Marquardt)

---

# تدفق البيانات (High-level Flow)

1. Env يُنشئ حالة أو يقدم أمثلة I/O إلى UnifiedAgent.
2. UnifiedAgent يُرسل الداتا إلى ProblemIdentifier.
3. ProblemIdentifier يردّ نوع اللغز مع درجة ثقة.
4. StrategySelector يبني خطة تنفيذ. إذا type==navigation → HierarchicalRLAgent. إذا reverse_engineering → LinearAlgebraSolver. إذا hybrid → مزيج منهما.
5. أثناء التنفيذ، Logger & Evaluator يجمع الأداء ويرجعه إلى CurriculumManager لتحديث مسارات التدريب.
6. بعد نجاح التدريب، النماذج تحفظ في ModelZoo وتُعرض عبر InterfaceLayer.

---

# خطة تدريب مقترحة (Training Plan)

1. **جمع/إنشاء بيانات أساسية**: 1000 أمثلة ملاحة بسيطة، 1000 عينات هندسة عكسية خطية.
2. **Baseline tests**: تشغيل UnifiedAgent على المجموعة لمعرفة الأداء الابتدائي.
3. **Curriculum phases**:
   - Phase A: stationary maps / linear functions
   - Phase B: add obstacles / add noise to I/O
   - Phase C: dynamic forbidden states / multiple targets / non-linear functions
4. **Metrics**:
   - Navigation: success_rate, avg_steps_to_target, policy_entropy
   - Reverse-engineering: coefficient_MSE, R^2, solve_time
   - System: end-to-end throughput, classification accuracy (ProblemIdentifier)
5. **Iterate**: تحديث السيناريوهات، إعادة تدريب، حفظ أفضل نماذج في ModelZoo.

---

# تجارب مقترحة (Experiments Matrix)

- E1: اختبار ProblemIdentifier على حالات مختلطة (precision/recall benchmark).
- E2: تدريب HierarchicalRLAgent على خرائط ثابتة ثم اختبار على خرائط ديناميكية.
- E3: مقارنة LinearAlgebraSolver (OLS vs Ridge vs SVD) تحت ضجيج متزايد.
- E4: عمل hybrid run عند غموض 0.4-0.6 وثم قياس الأداء النهائي.

---

# تخزين ومرجعية بيانات (Logging & Storage)

- سجلات التجارب بصيغة JSONL: تتضمن seed, scenario_spec, model_version, metrics.
- تخزين النماذج: ModelZoo بنمط semantic versioning.
- استخدام نظام تتبع تجارب (مثلاً: Weights & Biases أو MLflow).

---

# خيارات التنفيذ التقنية (Implementation Notes)

- تمثيل الخريطة:
  - بداية خفيفة: NetworkX (تنفيذ محلي، اختبارات)
  - إنتاج/توسعة: Neo4j أو ArangoDB (Graph DB) مع واجهة REST
- لغة التنفيذ: Python 3.10+، استخدام PyTorch للـ RL، NumPy/Scipy للـ solvers
- واجهات: REST API (FastAPI) + CLI

---

# واجهات برمجية مقترحة (API Endpoints)

- POST /solve -> payload: {scenario or io_examples} -> response: {solution, logs, model_version}
- POST /train -> payload: {phase, params} -> response: {job_id}
- GET /status/{job_id} -> job status
- GET /models -> list models in ModelZoo

---

# عناصر قابلية القياس (KPIs)

- دقة ProblemIdentifier >= 95% على بيانات validation (هدف نهائي)
- Navigation success_rate >= 90% لخرائط المرحلة C (طويل الأجل)
- Reverse-engineering R^2 >= 0.98 على معاملات خطية بدون ضجيج
- وقت استنتاج (inference) < 200 ms للحلول البسيطة

---

# خطوات العمل التالية (Next Actions)

1. توليد مجموعة بيانات أولية (ScenarioGenerator) — 1 أسبوع عمل.
2. تنفيذ واجهة ProblemIdentifier بسيطة (rule-based + classifier) — 3 أيام.
3. تنفيذ scaffold لـ HierarchicalRLAgent مع بيئة مُبسطة — 1 أسبوع.
4. تنفيذ LinearAlgebraSolver (SVD + ridge) — 2 أيام.
5. إعداد CI لتجارب التدريب وربط Logger بـ ModelZoo — 3 أيام.

---

# Checklist للنسخة الأولى (MVP)

- [x] تمثيل العقد الأساسية وتصميم العلاقات
- [ ] تنفيذ ProblemIdentifier (baseline)
- [ ] تنفيذ StrategySelector (baseline)
- [ ] HierarchicalRLAgent scaffold
- [ ] LinearAlgebraSolver
- [ ] ScenarioGenerator (basic)
- [ ] Logger & Evaluator + storage
- [ ] API layer (FastAPI)

---

# ملاحظة أخيرة

هذه الخريطة قابلة للتوسع. يمكنني الآن تحوِّيلها إلى:
- تمثيل بصيغة JSON-LD/GraphSON/Neo4j import
- كود scaffold (ملفات Python) للمكونات الأساسية
- رسم بياني تفاعلي (visualization)

اخبرني أيّ صيغة تفضّلها للخطوة التالية: تحويل إلى Graph DB import، أو شجرة Python modules scaffold، أو رسم تفاعلي؟

