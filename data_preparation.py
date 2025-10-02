
import json

def prepare_learning_data():
    learning_data = [
        {
            "concept": "التعلم المعزز (Reinforcement Learning)",
            "description": "فرع من التعلم الآلي يهتم بكيفية اتخاذ الوكلاء قرارات في بيئة معينة لتحقيق أقصى قدر من المكافآت المتراكمة.",
            "related_terms": ["Agent", "Environment", "Reward", "Policy", "State", "Action"]
        },
        {
            "concept": "التعلم المعزز الهرمي (HRL)",
            "description": "نهج يهدف إلى معالجة تعقيد المهام طويلة الأمد عبر تقسيمها إلى تسلسل هرمي من المهام الفرعية الأبسط.",
            "related_terms": ["Temporal Abstraction", "Subgoal", "Meta-Controller"]
        },
        {
            "concept": "الاستدلال العلائقي (Relational Reasoning)",
            "description": "القدرة على فهم العلاقات بين الكيانات المختلفة في البيئة بدلاً من التركيز على خصائصها الفردية.",
            "related_terms": ["Graph Neural Networks", "Relational Inductive Bias"]
        },
        {
            "concept": "تعلم التمثيل (Representation Learning)",
            "description": "اكتشاف تمثيلات فعالة للبيانات تسهل عملية التعلم والاستدلال، مما يسمح بفهم العالم على مستوى أعلى من التجريد.",
            "related_terms": ["Embedding Spaces", "Disentangled Representations"]
        },
        {
            "concept": "التعلم الفوقي (Meta-Learning)",
            "description": "\"التعلم للتعلم\"، وهو تصميم نماذج يمكنها تعلم كيفية التكيف بسرعة مع مهام جديدة باستخدام بيانات قليلة.",
            "related_terms": ["Few-Shot Learning", "Fast Adaptation", "MAML"]
        },
        {
            "concept": "التعلم التحويلي (Transfer Learning)",
            "description": "استخدام المعرفة المكتسبة من مهمة (المجال المصدر) لتحسين أداء النموذج في مهمة أخرى ذات صلة (المجال الهدف).",
            "related_terms": ["Domain Adaptation", "Knowledge Distillation"]
        },
        {
            "concept": "الاستدلال السببي (Causal Reasoning)",
            "description": "القدرة على فهم علاقات السبب والنتيجة بين الأحداث، بدلاً من مجرد الارتباطات، لاتخاذ قرارات أكثر قوة.",
            "related_terms": ["Counterfactual Reasoning", "Causal Graphs", "Do-calculus"]
        }
    ]
    with open('learning_data.jsonl', 'w', encoding='utf-8') as f:
        for entry in learning_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def prepare_puzzles_data():
    puzzles_data = [
        {
            "puzzle": "معضلة الاستكشاف مقابل الاستغلال",
            "domain": "التعلم المعزز",
            "description": "كيف يوازن الوكيل بين استكشاف أفعال جديدة قد تكون أفضل، واستغلال المعرفة الحالية لتحقيق أقصى مكافأة؟"
        },
        {
            "puzzle": "تصميم دالة المكافأة",
            "domain": "التعلم المعزز",
            "description": "كيف يمكن تصميم دالة مكافأة توصل الوكيل للسلوك المرغوب، خاصة عندما تكون المكافآت متفرقة أو متأخرة؟"
        },
        {
            "puzzle": "التعميم مقابل التخصيص",
            "domain": "تعلم التمثيل / التعلم المعزز",
            "description": "كيف يمكن بناء نموذج يعمم بشكل جيد على بيانات جديدة وغير مرئية دون أن يفقد دقته على بيانات التدريب (Overfitting)؟"
        },
        {
            "puzzle": "السببية مقابل الارتباط",
            "domain": "الاستدلال السببي / الفلسفة",
            "description": "كيف يمكن للنموذج التمييز بين علاقة سببية حقيقية ومجرد ارتباط إحصائي، لتجنب الاستنتاجات الخاطئة؟"
        },
        {
            "puzzle": "الفهم مقابل التنبؤ",
            "domain": "الفلسفة / تعلم التمثيل",
            "description": "هل أداء النموذج الجيد يعني أنه \"يفهم\" المهمة حقًا، أم أنه مجرد نظام متطور للتعرف على الأنماط؟"
        }
    ]
    with open('puzzles_data.jsonl', 'w', encoding='utf-8') as f:
        for entry in puzzles_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def prepare_data_source_data():
    data_source_data = [
        {
            "data_type": "بيانات غير مؤكدة أو صاخبة",
            "associated_challenge": "التعامل مع عدم اليقين (Handling Uncertainty)",
            "example": "بيانات من مستشعرات روبوتية قد تكون غير دقيقة."
        },
        {
            "data_type": "بيانات ذات أبعاد عالية",
            "associated_challenge": "لعنة الأبعاد (Curse of Dimensionality)",
            "example": "مدخلات صور خام (مثل ألعاب Atari) حيث كل بكسل هو بُعد."
        },
        {
            "data_type": "بيانات ذات علاقات معقدة",
            "associated_challenge": "تمثيل العلاقات (Representing Relations)",
            "example": "بيانات من شبكات اجتماعية أو جزيئات كيميائية."
        },
        {
            "data_type": "بيانات قليلة أو نادرة",
            "associated_challenge": "عدم كفاءة العينة (Sample Inefficiency)",
            "example": "سيناريوهات طبية حيث يكون جمع بيانات المرضى مكلفًا ونادرًا."
        },
        {
            "data_type": "بيانات من توزيعات مختلفة",
            "associated_challenge": "اختلاف توزيع البيانات (Data Distribution Mismatch)",
            "example": "نموذج مدرب على صور نهارية ويُطلب منه العمل على صور ليلية."
        }
    ]
    with open('data_source_data.jsonl', 'w', encoding='utf-8') as f:
        for entry in data_source_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def prepare_moves_data():
    moves_data = [
        {
            "move": "تحسين السياسة (Policy Optimization)",
            "type": "خوارزمية",
            "objective": "تعديل استراتيجية الوكيل لزيادة المكافأة المتوقعة."
        },
        {
            "move": "تقريب الدالة (Function Approximation)",
            "type": "تقنية",
            "objective": "استخدام الشبكات العصبية لتمثيل دوال القيمة أو السياسات في البيئات المعقدة."
        },
        {
            "move": "التجريد الزمني (Temporal Abstraction)",
            "type": "تقنية",
            "objective": "إنشاء \"حركات كبرى\" (Macro-Actions) تمتد لفترات زمنية طويلة."
        },
        {
            "move": "توليد الأهداف الفرعية (Subgoal Generation)",
            "type": "تقنية",
            "objective": "تقسيم مهمة معقدة إلى أهداف وسيطة أبسط وأكثر قابلية للإدارة."
        },
        {
            "move": "إعادة تشغيل الخبرة (Experience Replay)",
            "type": "تقنية",
            "objective": "تخزين التجارب وإعادة استخدامها لتدريب أكثر كفاءة واستقرارًا."
        },
        {
            "move": "حساب التدخل (Do-calculus)",
            "type": "أداة رياضية",
            "objective": "حساب التأثير السببي لتدخل معين في النظام."
        }
    ]
    with open('moves_data.jsonl', 'w', encoding='utf-8') as f:
        for entry in moves_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def prepare_relationships_data():
    relationships_data = [
        {
            "element_a": "عدم كفاءة العينة (RL)",
            "element_b": "التعلم الفوقي / التحويلي",
            "relationship": "B هو حل محتمل لـ A، حيث يهدف إلى تسريع التعلم من بيانات أقل."
        },
        {
            "element_a": "نظرية التحسين (Math)",
            "element_b": "تدريب الشبكات العصبية (RL)",
            "relationship": "A هي الأداة الأساسية لتنفيذ B (مثل استخدام الانحدار التدرجي)."
        },
        {
            "element_a": "الاستدلال العلائقي (Reasoning)",
            "element_b": "التعلم الهرمي (HRL)",
            "relationship": "A يمكن أن يعزز B عبر تمكين فهم أفضل للعلاقات بين الأهداف الفرعية."
        },
        {
            "element_a": "قابلية التوسع (Graphs)",
            "element_b": "الاستدلال العلائقي (Reasoning)",
            "relationship": "تحديات A تؤثر بشكل مباشر على قابلية تطبيق B في المشاكل الكبيرة."
        },
        {
            "element_a": "الأسئلة الفلسفية (Philosophy)",
            "element_b": "جميع التحديات التقنية",
            "relationship": "A توفر الإطار الذي يوجه البحث والأولويات لحل B."
        }
    ]
    with open('relationships_data.jsonl', 'w', encoding='utf-8') as f:
        for entry in relationships_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def prepare_challenges_data():
    challenges_data = [
        {
            "challenge": "عدم كفاءة العينة",
            "domain": "التعلم المعزز",
            "description": "الحاجة إلى كميات هائلة من البيانات للتعلم بفعالية."
        },
        {
            "challenge": "عدم استقرار التدريب",
            "domain": "التعلم المعزز العميق",
            "description": "حساسية شديدة للمعاملات الفائقة وصعوبة في إعادة إنتاج النتائج."
        },
        {
            "challenge": "التعميم الضعيف",
            "domain": "التعلم المعزز / التمثيل",
            "description": "فشل النموذج في التكيف مع بيئات جديدة تختلف قليلاً عن بيئة التدريب."
        },
        {
            "challenge": "تحديد الهيكل الهرمي",
            "domain": "التعلم المعزز الهرمي",
            "description": "صعوبة تحديد المستويات الصحيحة من التجريد والمهام الفرعية."
        },
        {
            "challenge": "قابلية التفسير",
            "domain": "تعلم التمثيل",
            "description": "صعوبة فهم كيف ولماذا اتخذ النموذج قرارًا معينًا."
        },
        {
            "challenge": "الحد الأدنى المحلي والنقاط السرجية",
            "domain": "نظرية التحسين",
            "description": "إمكانية أن تتعثر خوارزميات التحسين في حلول دون المستوى الأمثل."
        },
        {
            "challenge": "النقل السلبي (Negative Transfer)",
            "domain": "التعلم التحويلي",
            "description": "تدهور الأداء عند نقل المعرفة من مجال مصدر غير مناسب."
        },
        {
            "challenge": "اكتشاف العلاقات السببية",
            "domain": "الاستدلال السببي",
            "description": "صعوبة التمييز بين الارتباط والسببية من البيانات الرصدية وحدها."
        }
    ]
    with open('challenges_data.jsonl', 'w', encoding='utf-8') as f:
        for entry in challenges_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def prepare_strategies_data():
    strategies_data = [
        {
            "strategy": "استراتيجية \"تعلم لتتعلم\"",
            "core_principle": "استخدام التعلم الفوقي للتكيف السريع مع المهام الجديدة.",
            "targeted_challenges": ["عدم كفاءة العينة", "التعميم الضعيف بين المهام"]
        },
        {
            "strategy": "استراتيجية \"قف على أكتاف العمالقة\"",
            "core_principle": "استخدام التعلم التحويلي للاستفادة من النماذج المدربة مسبقًا.",
            "targeted_challenges": ["عدم كفاءة العينة", "الحاجة إلى بيانات ضخمة"]
        },
        {
            "strategy": "استراتيجية \"فرّق تَسُد\"",
            "core_principle": "استخدام التعلم الهرمي لتقسيم المشاكل المعقدة.",
            "targeted_challenges": ["الأبعاد العالية", "تخطيط طويل المدى", "تصميم المكافآت المتفرقة"]
        },
        {
            "strategy": "استراتيجية \"ابحث عن السبب\"",
            "core_principle": "دمج الاستدلال السببي لفهم أعمق للنظام.",
            "targeted_challenges": ["التعميم خارج نطاق التدريب", "قابلية التفسير", "القرارات غير العادلة"]
        },
        {
            "strategy": "استراتيجية \"ابنِ علاقات\"",
            "core_principle": "استخدام تعلم التمثيل العلائقي لفهم الروابط بين الكيانات.",
            "targeted_challenges": ["البيئات المعقدة ذات الكيانات المتعددة", "التعميم الهيكلي"]
        }
    ]
    with open('strategies_data.jsonl', 'w', encoding='utf-8') as f:
        for entry in strategies_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    prepare_learning_data()
    prepare_puzzles_data()
    prepare_data_source_data()
    prepare_moves_data()
    prepare_relationships_data()
    prepare_challenges_data()
    prepare_strategies_data()




import csv

def write_to_csv(data, filename, fieldnames):
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def write_to_txt(data, filename, fields):
    with open(filename, 'w', encoding='utf-8') as f:
        for entry in data:
            for field in fields:
                f.write(f"{field}: {entry.get(field, '')}\n")
            f.write("\n")

def prepare_learning_data_csv():
    learning_data = [
        {
            "concept": "التعلم المعزز (Reinforcement Learning)",
            "description": "فرع من التعلم الآلي يهتم بكيفية اتخاذ الوكلاء قرارات في بيئة معينة لتحقيق أقصى قدر من المكافآت المتراكمة.",
            "related_terms": "Agent, Environment, Reward, Policy, State, Action"
        },
        {
            "concept": "التعلم المعزز الهرمي (HRL)",
            "description": "نهج يهدف إلى معالجة تعقيد المهام طويلة الأمد عبر تقسيمها إلى تسلسل هرمي من المهام الفرعية الأبسط.",
            "related_terms": "Temporal Abstraction, Subgoal, Meta-Controller"
        },
        {
            "concept": "الاستدلال العلائقي (Relational Reasoning)",
            "description": "القدرة على فهم العلاقات بين الكيانات المختلفة في البيئة بدلاً من التركيز على خصائصها الفردية.",
            "related_terms": "Graph Neural Networks, Relational Inductive Bias"
        },
        {
            "concept": "تعلم التمثيل (Representation Learning)",
            "description": "اكتشاف تمثيلات فعالة للبيانات تسهل عملية التعلم والاستدلال، مما يسمح بفهم العالم على مستوى أعلى من التجريد.",
            "related_terms": "Embedding Spaces, Disentangled Representations"
        },
        {
            "concept": "التعلم الفوقي (Meta-Learning)",
            "description": "\"التعلم للتعلم\"، وهو تصميم نماذج يمكنها تعلم كيفية التكيف بسرعة مع مهام جديدة باستخدام بيانات قليلة.",
            "related_terms": "Few-Shot Learning, Fast Adaptation, MAML"
        },
        {
            "concept": "التعلم التحويلي (Transfer Learning)",
            "description": "استخدام المعرفة المكتسبة من مهمة (المجال المصدر) لتحسين أداء النموذج في مهمة أخرى ذات صلة (المجال الهدف).",
            "related_terms": "Domain Adaptation, Knowledge Distillation"
        },
        {
            "concept": "الاستدلال السببي (Causal Reasoning)",
            "description": "القدرة على فهم علاقات السبب والنتيجة بين الأحداث، بدلاً من مجرد الارتباطات، لاتخاذ قرارات أكثر قوة.",
            "related_terms": "Counterfactual Reasoning, Causal Graphs, Do-calculus"
        }
    ]
    write_to_csv(learning_data, 'learning_data.csv', ['concept', 'description', 'related_terms'])

def prepare_puzzles_data_csv():
    puzzles_data = [
        {
            "puzzle": "معضلة الاستكشاف مقابل الاستغلال",
            "domain": "التعلم المعزز",
            "description": "كيف يوازن الوكيل بين استكشاف أفعال جديدة قد تكون أفضل، واستغلال المعرفة الحالية لتحقيق أقصى مكافأة؟"
        },
        {
            "puzzle": "تصميم دالة المكافأة",
            "domain": "التعلم المعزز",
            "description": "كيف يمكن تصميم دالة مكافأة توصل الوكيل للسلوك المرغوب، خاصة عندما تكون المكافآت متفرقة أو متأخرة؟"
        },
        {
            "puzzle": "التعميم مقابل التخصيص",
            "domain": "تعلم التمثيل / التعلم المعزز",
            "description": "كيف يمكن بناء نموذج يعمم بشكل جيد على بيانات جديدة وغير مرئية دون أن يفقد دقته على بيانات التدريب (Overfitting)؟"
        },
        {
            "puzzle": "السببية مقابل الارتباط",
            "domain": "الاستدلال السببي / الفلسفة",
            "description": "كيف يمكن للنموذج التمييز بين علاقة سببية حقيقية ومجرد ارتباط إحصائي، لتجنب الاستنتاجات الخاطئة؟"
        },
        {
            "puzzle": "الفهم مقابل التنبؤ",
            "domain": "الفلسفة / تعلم التمثيل",
            "description": "هل أداء النموذج الجيد يعني أنه \"يفهم\" المهمة حقًا، أم أنه مجرد نظام متطور للتعرف على الأنماط؟"
        }
    ]
    write_to_csv(puzzles_data, 'puzzles_data.csv', ['puzzle', 'domain', 'description'])

def prepare_data_source_data_csv():
    data_source_data = [
        {
            "data_type": "بيانات غير مؤكدة أو صاخبة",
            "associated_challenge": "التعامل مع عدم اليقين (Handling Uncertainty)",
            "example": "بيانات من مستشعرات روبوتية قد تكون غير دقيقة."
        },
        {
            "data_type": "بيانات ذات أبعاد عالية",
            "associated_challenge": "لعنة الأبعاد (Curse of Dimensionality)",
            "example": "مدخلات صور خام (مثل ألعاب Atari) حيث كل بكسل هو بُعد."
        },
        {
            "data_type": "بيانات ذات علاقات معقدة",
            "associated_challenge": "تمثيل العلاقات (Representing Relations)",
            "example": "بيانات من شبكات اجتماعية أو جزيئات كيميائية."
        },
        {
            "data_type": "بيانات قليلة أو نادرة",
            "associated_challenge": "عدم كفاءة العينة (Sample Inefficiency)",
            "example": "سيناريوهات طبية حيث يكون جمع بيانات المرضى مكلفًا ونادرًا."
        },
        {
            "data_type": "بيانات من توزيعات مختلفة",
            "associated_challenge": "اختلاف توزيع البيانات (Data Distribution Mismatch)",
            "example": "نموذج مدرب على صور نهارية ويُطلب منه العمل على صور ليلية."
        }
    ]
    write_to_csv(data_source_data, 'data_source_data.csv', ['data_type', 'associated_challenge', 'example'])

def prepare_moves_data_csv():
    moves_data = [
        {
            "move": "تحسين السياسة (Policy Optimization)",
            "type": "خوارزمية",
            "objective": "تعديل استراتيجية الوكيل لزيادة المكافأة المتوقعة."
        },
        {
            "move": "تقريب الدالة (Function Approximation)",
            "type": "تقنية",
            "objective": "استخدام الشبكات العصبية لتمثيل دوال القيمة أو السياسات في البيئات المعقدة."
        },
        {
            "move": "التجريد الزمني (Temporal Abstraction)",
            "type": "تقنية",
            "objective": "إنشاء \"حركات كبرى\" (Macro-Actions) تمتد لفترات زمنية طويلة."
        },
        {
            "move": "توليد الأهداف الفرعية (Subgoal Generation)",
            "type": "تقنية",
            "objective": "تقسيم مهمة معقدة إلى أهداف وسيطة أبسط وأكثر قابلية للإدارة."
        },
        {
            "move": "إعادة تشغيل الخبرة (Experience Replay)",
            "type": "تقنية",
            "objective": "تخزين التجارب وإعادة استخدامها لتدريب أكثر كفاءة واستقرارًا."
        },
        {
            "move": "حساب التدخل (Do-calculus)",
            "type": "أداة رياضية",
            "objective": "حساب التأثير السببي لتدخل معين في النظام."
        }
    ]
    write_to_csv(moves_data, 'moves_data.csv', ['move', 'type', 'objective'])

def prepare_relationships_data_csv():
    relationships_data = [
        {
            "element_a": "عدم كفاءة العينة (RL)",
            "element_b": "التعلم الفوقي / التحويلي",
            "relationship": "B هو حل محتمل لـ A، حيث يهدف إلى تسريع التعلم من بيانات أقل."
        },
        {
            "element_a": "نظرية التحسين (Math)",
            "element_b": "تدريب الشبكات العصبية (RL)",
            "relationship": "A هي الأداة الأساسية لتنفيذ B (مثل استخدام الانحدار التدرجي)."
        },
        {
            "element_a": "الاستدلال العلائقي (Reasoning)",
            "element_b": "التعلم الهرمي (HRL)",
            "relationship": "A يمكن أن يعزز B عبر تمكين فهم أفضل للعلاقات بين الأهداف الفرعية."
        },
        {
            "element_a": "قابلية التوسع (Graphs)",
            "element_b": "الاستدلال العلائقي (Reasoning)",
            "relationship": "تحديات A تؤثر بشكل مباشر على قابلية تطبيق B في المشاكل الكبيرة."
        },
        {
            "element_a": "الأسئلة الفلسفية (Philosophy)",
            "element_b": "جميع التحديات التقنية",
            "relationship": "A توفر الإطار الذي يوجه البحث والأولويات لحل B."
        }
    ]
    write_to_csv(relationships_data, 'relationships_data.csv', ['element_a', 'element_b', 'relationship'])

def prepare_challenges_data_csv():
    challenges_data = [
        {
            "challenge": "عدم كفاءة العينة",
            "domain": "التعلم المعزز",
            "description": "الحاجة إلى كميات هائلة من البيانات للتعلم بفعالية."
        },
        {
            "challenge": "عدم استقرار التدريب",
            "domain": "التعلم المعزز العميق",
            "description": "حساسية شديدة للمعاملات الفائقة وصعوبة في إعادة إنتاج النتائج."
        },
        {
            "challenge": "التعميم الضعيف",
            "domain": "التعلم المعزز / التمثيل",
            "description": "فشل النموذج في التكيف مع بيئات جديدة تختلف قليلاً عن بيئة التدريب."
        },
        {
            "challenge": "تحديد الهيكل الهرمي",
            "domain": "التعلم المعزز الهرمي",
            "description": "صعوبة تحديد المستويات الصحيحة من التجريد والمهام الفرعية."
        },
        {
            "challenge": "قابلية التفسير",
            "domain": "تعلم التمثيل",
            "description": "صعوبة فهم كيف ولماذا اتخذ النموذج قرارًا معينًا."
        },
        {
            "challenge": "الحد الأدنى المحلي والنقاط السرجية",
            "domain": "نظرية التحسين",
            "description": "إمكانية أن تتعثر خوارزميات التحسين في حلول دون المستوى الأمثل."
        },
        {
            "challenge": "النقل السلبي (Negative Transfer)",
            "domain": "التعلم التحويلي",
            "description": "تدهور الأداء عند نقل المعرفة من مجال مصدر غير مناسب."
        },
        {
            "challenge": "اكتشاف العلاقات السببية",
            "domain": "الاستدلال السببي",
            "description": "صعوبة التمييز بين الارتباط والسببية من البيانات الرصدية وحدها."
        }
    ]
    write_to_csv(challenges_data, 'challenges_data.csv', ['challenge', 'domain', 'description'])

def prepare_strategies_data_csv():
    strategies_data = [
        {
            "strategy": "استراتيجية \"تعلم لتتعلم\"",
            "core_principle": "استخدام التعلم الفوقي للتكيف السريع مع المهام الجديدة.",
            "targeted_challenges": "عدم كفاءة العينة, التعميم الضعيف بين المهام"
        },
        {
            "strategy": "استراتيجية \"قف على أكتاف العمالقة\"",
            "core_principle": "استخدام التعلم التحويلي للاستفادة من النماذج المدربة مسبقًا.",
            "targeted_challenges": "عدم كفاءة العينة, الحاجة إلى بيانات ضخمة"
        },
        {
            "strategy": "استراتيجية \"فرّق تَسُد\"",
            "core_principle": "استخدام التعلم الهرمي لتقسيم المشاكل المعقدة.",
            "targeted_challenges": "الأبعاد العالية, تخطيط طويل المدى, تصميم المكافآت المتفرقة"
        },
        {
            "strategy": "استراتيجية \"ابحث عن السبب\"",
            "core_principle": "دمج الاستدلال السببي لفهم أعمق للنظام.",
            "targeted_challenges": "التعميم خارج نطاق التدريب, قابلية التفسير, القرارات غير العادلة"
        },
        {
            "strategy": "استراتيجية \"ابنِ علاقات\"",
            "core_principle": "استخدام تعلم التمثيل العلائقي لفهم الروابط بين الكيانات.",
            "targeted_challenges": "البيئات المعقدة ذات الكيانات المتعددة, التعميم الهيكلي"
        }
    ]
    write_to_csv(strategies_data, 'strategies_data.csv', ['strategy', 'core_principle', 'targeted_challenges'])

def prepare_learning_data_txt():
    learning_data = [
        {
            "concept": "التعلم المعزز (Reinforcement Learning)",
            "description": "فرع من التعلم الآلي يهتم بكيفية اتخاذ الوكلاء قرارات في بيئة معينة لتحقيق أقصى قدر من المكافآت المتراكمة.",
            "related_terms": "Agent, Environment, Reward, Policy, State, Action"
        },
        {
            "concept": "التعلم المعزز الهرمي (HRL)",
            "description": "نهج يهدف إلى معالجة تعقيد المهام طويلة الأمد عبر تقسيمها إلى تسلسل هرمي من المهام الفرعية الأبسط.",
            "related_terms": "Temporal Abstraction, Subgoal, Meta-Controller"
        },
        {
            "concept": "الاستدلال العلائقي (Relational Reasoning)",
            "description": "القدرة على فهم العلاقات بين الكيانات المختلفة في البيئة بدلاً من التركيز على خصائصها الفردية.",
            "related_terms": "Graph Neural Networks, Relational Inductive Bias"
        },
        {
            "concept": "تعلم التمثيل (Representation Learning)",
            "description": "اكتشاف تمثيلات فعالة للبيانات تسهل عملية التعلم والاستدلال، مما يسمح بفهم العالم على مستوى أعلى من التجريد.",
            "related_terms": "Embedding Spaces, Disentangled Representations"
        },
        {
            "concept": "التعلم الفوقي (Meta-Learning)",
            "description": "\"التعلم للتعلم\"، وهو تصميم نماذج يمكنها تعلم كيفية التكيف بسرعة مع مهام جديدة باستخدام بيانات قليلة.",
            "related_terms": "Few-Shot Learning, Fast Adaptation, MAML"
        },
        {
            "concept": "التعلم التحويلي (Transfer Learning)",
            "description": "استخدام المعرفة المكتسبة من مهمة (المجال المصدر) لتحسين أداء النموذج في مهمة أخرى ذات صلة (المجال الهدف).",
            "related_terms": "Domain Adaptation, Knowledge Distillation"
        },
        {
            "concept": "الاستدلال السببي (Causal Reasoning)",
            "description": "القدرة على فهم علاقات السبب والنتيجة بين الأحداث، بدلاً من مجرد الارتباطات، لاتخاذ قرارات أكثر قوة.",
            "related_terms": "Counterfactual Reasoning, Causal Graphs, Do-calculus"
        }
    ]
    write_to_txt(learning_data, 'learning_data.txt', ['concept', 'description', 'related_terms'])

def prepare_puzzles_data_txt():
    puzzles_data = [
        {
            "puzzle": "معضلة الاستكشاف مقابل الاستغلال",
            "domain": "التعلم المعزز",
            "description": "كيف يوازن الوكيل بين استكشاف أفعال جديدة قد تكون أفضل، واستغلال المعرفة الحالية لتحقيق أقصى مكافأة؟"
        },
        {
            "puzzle": "تصميم دالة المكافأة",
            "domain": "التعلم المعزز",
            "description": "كيف يمكن تصميم دالة مكافأة توصل الوكيل للسلوك المرغوب، خاصة عندما تكون المكافآت متفرقة أو متأخرة؟"
        },
        {
            "puzzle": "التعميم مقابل التخصيص",
            "domain": "تعلم التمثيل / التعلم المعزز",
            "description": "كيف يمكن بناء نموذج يعمم بشكل جيد على بيانات جديدة وغير مرئية دون أن يفقد دقته على بيانات التدريب (Overfitting)؟"
        },
        {
            "puzzle": "السببية مقابل الارتباط",
            "domain": "الاستدلال السببي / الفلسفة",
            "description": "كيف يمكن للنموذج التمييز بين علاقة سببية حقيقية ومجرد ارتباط إحصائي، لتجنب الاستنتاجات الخاطئة؟"
        },
        {
            "puzzle": "الفهم مقابل التنبؤ",
            "domain": "الفلسفة / تعلم التمثيل",
            "description": "هل أداء النموذج الجيد يعني أنه \"يفهم\" المهمة حقًا، أم أنه مجرد نظام متطور للتعرف على الأنماط؟"
        }
    ]
    write_to_txt(puzzles_data, 'puzzles_data.txt', ['puzzle', 'domain', 'description'])

def prepare_data_source_data_txt():
    data_source_data = [
        {
            "data_type": "بيانات غير مؤكدة أو صاخبة",
            "associated_challenge": "التعامل مع عدم اليقين (Handling Uncertainty)",
            "example": "بيانات من مستشعرات روبوتية قد تكون غير دقيقة."
        },
        {
            "data_type": "بيانات ذات أبعاد عالية",
            "associated_challenge": "لعنة الأبعاد (Curse of Dimensionality)",
            "example": "مدخلات صور خام (مثل ألعاب Atari) حيث كل بكسل هو بُعد."
        },
        {
            "data_type": "بيانات ذات علاقات معقدة",
            "associated_challenge": "تمثيل العلاقات (Representing Relations)",
            "example": "بيانات من شبكات اجتماعية أو جزيئات كيميائية."
        },
        {
            "data_type": "بيانات قليلة أو نادرة",
            "associated_challenge": "عدم كفاءة العينة (Sample Inefficiency)",
            "example": "سيناريوهات طبية حيث يكون جمع بيانات المرضى مكلفًا ونادرًا."
        },
        {
            "data_type": "بيانات من توزيعات مختلفة",
            "associated_challenge": "اختلاف توزيع البيانات (Data Distribution Mismatch)",
            "example": "نموذج مدرب على صور نهارية ويُطلب منه العمل على صور ليلية."
        }
    ]
    write_to_txt(data_source_data, 'data_source_data.txt', ['data_type', 'associated_challenge', 'example'])

def prepare_moves_data_txt():
    moves_data = [
        {
            "move": "تحسين السياسة (Policy Optimization)",
            "type": "خوارزمية",
            "objective": "تعديل استراتيجية الوكيل لزيادة المكافأة المتوقعة."
        },
        {
            "move": "تقريب الدالة (Function Approximation)",
            "type": "تقنية",
            "objective": "استخدام الشبكات العصبية لتمثيل دوال القيمة أو السياسات في البيئات المعقدة."
        },
        {
            "move": "التجريد الزمني (Temporal Abstraction)",
            "type": "تقنية",
            "objective": "إنشاء \"حركات كبرى\" (Macro-Actions) تمتد لفترات زمنية طويلة."
        },
        {
            "move": "توليد الأهداف الفرعية (Subgoal Generation)",
            "type": "تقنية",
            "objective": "تقسيم مهمة معقدة إلى أهداف وسيطة أبسط وأكثر قابلية للإدارة."
        },
        {
            "move": "إعادة تشغيل الخبرة (Experience Replay)",
            "type": "تقنية",
            "objective": "تخزين التجارب وإعادة استخدامها لتدريب أكثر كفاءة واستقرارًا."
        },
        {
            "move": "حساب التدخل (Do-calculus)",
            "type": "أداة رياضية",
            "objective": "حساب التأثير السببي لتدخل معين في النظام."
        }
    ]
    write_to_txt(moves_data, 'moves_data.txt', ['move', 'type', 'objective'])

def prepare_relationships_data_txt():
    relationships_data = [
        {
            "element_a": "عدم كفاءة العينة (RL)",
            "element_b": "التعلم الفوقي / التحويلي",
            "relationship": "B هو حل محتمل لـ A، حيث يهدف إلى تسريع التعلم من بيانات أقل."
        },
        {
            "element_a": "نظرية التحسين (Math)",
            "element_b": "تدريب الشبكات العصبية (RL)",
            "relationship": "A هي الأداة الأساسية لتنفيذ B (مثل استخدام الانحدار التدرجي)."
        },
        {
            "element_a": "الاستدلال العلائقي (Reasoning)",
            "element_b": "التعلم الهرمي (HRL)",
            "relationship": "A يمكن أن يعزز B عبر تمكين فهم أفضل للعلاقات بين الأهداف الفرعية."
        },
        {
            "element_a": "قابلية التوسع (Graphs)",
            "element_b": "الاستدلال العلائقي (Reasoning)",
            "relationship": "تحديات A تؤثر بشكل مباشر على قابلية تطبيق B في المشاكل الكبيرة."
        },
        {
            "element_a": "الأسئلة الفلسفية (Philosophy)",
            "element_b": "جميع التحديات التقنية",
            "relationship": "A توفر الإطار الذي يوجه البحث والأولويات لحل B."
        }
    ]
    write_to_txt(relationships_data, 'relationships_data.txt', ['element_a', 'element_b', 'relationship'])

def prepare_challenges_data_txt():
    challenges_data = [
        {
            "challenge": "عدم كفاءة العينة",
            "domain": "التعلم المعزز",
            "description": "الحاجة إلى كميات هائلة من البيانات للتعلم بفعالية."
        },
        {
            "challenge": "عدم استقرار التدريب",
            "domain": "التعلم المعزز العميق",
            "description": "حساسية شديدة للمعاملات الفائقة وصعوبة في إعادة إنتاج النتائج."
        },
        {
            "challenge": "التعميم الضعيف",
            "domain": "التعلم المعزز / التمثيل",
            "description": "فشل النموذج في التكيف مع بيئات جديدة تختلف قليلاً عن بيئة التدريب."
        },
        {
            "challenge": "تحديد الهيكل الهرمي",
            "domain": "التعلم المعزز الهرمي",
            "description": "صعوبة تحديد المستويات الصحيحة من التجريد والمهام الفرعية."
        },
        {
            "challenge": "قابلية التفسير",
            "domain": "تعلم التمثيل",
            "description": "صعوبة فهم كيف ولماذا اتخذ النموذج قرارًا معينًا."
        },
        {
            "challenge": "الحد الأدنى المحلي والنقاط السرجية",
            "domain": "نظرية التحسين",
            "description": "إمكانية أن تتعثر خوارزميات التحسين في حلول دون المستوى الأمثل."
        },
        {
            "challenge": "النقل السلبي (Negative Transfer)",
            "domain": "التعلم التحويلي",
            "description": "تدهور الأداء عند نقل المعرفة من مجال مصدر غير مناسب."
        },
        {
            "challenge": "اكتشاف العلاقات السببية",
            "domain": "الاستدلال السببي",
            "description": "صعوبة التمييز بين الارتباط والسببية من البيانات الرصدية وحدها."
        }
    ]
    write_to_txt(challenges_data, 'challenges_data.txt', ['challenge', 'domain', 'description'])

def prepare_strategies_data_txt():
    strategies_data = [
        {
            "strategy": "استراتيجية \"تعلم لتتعلم\"",
            "core_principle": "استخدام التعلم الفوقي للتكيف السريع مع المهام الجديدة.",
            "targeted_challenges": "عدم كفاءة العينة, التعميم الضعيف بين المهام"
        },
        {
            "strategy": "استراتيجية \"قف على أكتاف العمالقة\"",
            "core_principle": "استخدام التعلم التحويلي للاستفادة من النماذج المدربة مسبقًا.",
            "targeted_challenges": "عدم كفاءة العينة, الحاجة إلى بيانات ضخمة"
        },
        {
            "strategy": "استراتيجية \"فرّق تَسُد\"",
            "core_principle": "استخدام التعلم الهرمي لتقسيم المشاكل المعقدة.",
            "targeted_challenges": "الأبعاد العالية, تخطيط طويل المدى, تصميم المكافآت المتفرقة"
        },
        {
            "strategy": "استراتيجية \"ابحث عن السبب\"",
            "core_principle": "دمج الاستدلال السببي لفهم أعمق للنظام.",
            "targeted_challenges": "التعميم خارج نطاق التدريب, قابلية التفسير, القرارات غير العادلة"
        },
        {
            "strategy": "استراتيجية \"ابنِ علاقات\"",
            "core_principle": "استخدام تعلم التمثيل العلائقي لفهم الروابط بين الكيانات.",
            "targeted_challenges": "البيئات المعقدة ذات الكيانات المتعددة, التعميم الهيكلي"
        }
    ]
    write_to_txt(strategies_data, 'strategies_data.txt', ['strategy', 'core_principle', 'targeted_challenges'])

if __name__ == '__main__':
    prepare_learning_data()
    prepare_puzzles_data()
    prepare_data_source_data()
    prepare_moves_data()
    prepare_relationships_data()
    prepare_challenges_data()
    prepare_strategies_data()

    prepare_learning_data_csv()
    prepare_puzzles_data_csv()
    prepare_data_source_data_csv()
    prepare_moves_data_csv()
    prepare_relationships_data_csv()
    prepare_challenges_data_csv()
    prepare_strategies_data_csv()

    prepare_learning_data_txt()
    prepare_puzzles_data_txt()
    prepare_data_source_data_txt()
    prepare_moves_data_txt()
    prepare_relationships_data_txt()
    prepare_challenges_data_txt()
    prepare_strategies_data_txt()




import random

def split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    random.shuffle(data)
    total_size = len(data)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    return train_data, val_data, test_data

def prepare_split_data():
    # Learning Data
    learning_data = [
        {
            "concept": "التعلم المعزز (Reinforcement Learning)",
            "description": "فرع من التعلم الآلي يهتم بكيفية اتخاذ الوكلاء قرارات في بيئة معينة لتحقيق أقصى قدر من المكافآت المتراكمة.",
            "related_terms": ["Agent", "Environment", "Reward", "Policy", "State", "Action"]
        },
        {
            "concept": "التعلم المعزز الهرمي (HRL)",
            "description": "نهج يهدف إلى معالجة تعقيد المهام طويلة الأمد عبر تقسيمها إلى تسلسل هرمي من المهام الفرعية الأبسط.",
            "related_terms": ["Temporal Abstraction", "Subgoal", "Meta-Controller"]
        },
        {
            "concept": "الاستدلال العلائقي (Relational Reasoning)",
            "description": "القدرة على فهم العلاقات بين الكيانات المختلفة في البيئة بدلاً من التركيز على خصائصها الفردية.",
            "related_terms": ["Graph Neural Networks", "Relational Inductive Bias"]
        },
        {
            "concept": "تعلم التمثيل (Representation Learning)",
            "description": "اكتشاف تمثيلات فعالة للبيانات تسهل عملية التعلم والاستدلال، مما يسمح بفهم العالم على مستوى أعلى من التجريد.",
            "related_terms": ["Embedding Spaces", "Disentangled Representations"]
        },
        {
            "concept": "التعلم الفوقي (Meta-Learning)",
            "description": "\"التعلم للتعلم\"، وهو تصميم نماذج يمكنها تعلم كيفية التكيف بسرعة مع مهام جديدة باستخدام بيانات قليلة.",
            "related_terms": ["Few-Shot Learning", "Fast Adaptation", "MAML"]
        },
        {
            "concept": "التعلم التحويلي (Transfer Learning)",
            "description": "استخدام المعرفة المكتسبة من مهمة (المجال المصدر) لتحسين أداء النموذج في مهمة أخرى ذات صلة (المجال الهدف).",
            "related_terms": ["Domain Adaptation", "Knowledge Distillation"]
        },
        {
            "concept": "الاستدلال السببي (Causal Reasoning)",
            "description": "القدرة على فهم علاقات السبب والنتيجة بين الأحداث، بدلاً من مجرد الارتباطات، لاتخاذ قرارات أكثر قوة.",
            "related_terms": ["Counterfactual Reasoning", "Causal Graphs", "Do-calculus"]
        }
    ]
    train_learning, val_learning, test_learning = split_data(learning_data)
    with open("learning_train.jsonl", "w", encoding="utf-8") as f:
        for entry in train_learning:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    with open("learning_val.jsonl", "w", encoding="utf-8") as f:
        for entry in val_learning:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    with open("learning_test.jsonl", "w", encoding="utf-8") as f:
        for entry in test_learning:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Puzzles Data
    puzzles_data = [
        {
            "puzzle": "معضلة الاستكشاف مقابل الاستغلال",
            "domain": "التعلم المعزز",
            "description": "كيف يوازن الوكيل بين استكشاف أفعال جديدة قد تكون أفضل، واستغلال المعرفة الحالية لتحقيق أقصى مكافأة؟"
        },
        {
            "puzzle": "تصميم دالة المكافأة",
            "domain": "التعلم المعزز",
            "description": "كيف يمكن تصميم دالة مكافأة توصل الوكيل للسلوك المرغوب، خاصة عندما تكون المكافآت متفرقة أو متأخرة؟"
        },
        {
            "puzzle": "التعميم مقابل التخصيص",
            "domain": "تعلم التمثيل / التعلم المعزز",
            "description": "كيف يمكن بناء نموذج يعمم بشكل جيد على بيانات جديدة وغير مرئية دون أن يفقد دقته على بيانات التدريب (Overfitting)؟"
        },
        {
            "puzzle": "السببية مقابل الارتباط",
            "domain": "الاستدلال السببي / الفلسفة",
            "description": "كيف يمكن للنموذج التمييز بين علاقة سببية حقيقية ومجرد ارتباط إحصائي، لتجنب الاستنتاجات الخاطئة؟"
        },
        {
            "puzzle": "الفهم مقابل التنبؤ",
            "domain": "الفلسفة / تعلم التمثيل",
            "description": "هل أداء النموذج الجيد يعني أنه \"يفهم\" المهمة حقًا، أم أنه مجرد نظام متطور للتعرف على الأنماط؟"
        }
    ]
    train_puzzles, val_puzzles, test_puzzles = split_data(puzzles_data)
    with open("puzzles_train.jsonl", "w", encoding="utf-8") as f:
        for entry in train_puzzles:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    with open("puzzles_val.jsonl", "w", encoding="utf-8") as f:
        for entry in val_puzzles:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    with open("puzzles_test.jsonl", "w", encoding="utf-8") as f:
        for entry in test_puzzles:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Data Source Data
    data_source_data = [
        {
            "data_type": "بيانات غير مؤكدة أو صاخبة",
            "associated_challenge": "التعامل مع عدم اليقين (Handling Uncertainty)",
            "example": "بيانات من مستشعرات روبوتية قد تكون غير دقيقة."
        },
        {
            "data_type": "بيانات ذات أبعاد عالية",
            "associated_challenge": "لعنة الأبعاد (Curse of Dimensionality)",
            "example": "مدخلات صور خام (مثل ألعاب Atari) حيث كل بكسل هو بُعد."
        },
        {
            "data_type": "بيانات ذات علاقات معقدة",
            "associated_challenge": "تمثيل العلاقات (Representing Relations)",
            "example": "بيانات من شبكات اجتماعية أو جزيئات كيميائية."
        },
        {
            "data_type": "بيانات قليلة أو نادرة",
            "associated_challenge": "عدم كفاءة العينة (Sample Inefficiency)",
            "example": "سيناريوهات طبية حيث يكون جمع بيانات المرضى مكلفًا ونادرًا."
        },
        {
            "data_type": "بيانات من توزيعات مختلفة",
            "associated_challenge": "اختلاف توزيع البيانات (Data Distribution Mismatch)",
            "example": "نموذج مدرب على صور نهارية ويُطلب منه العمل على صور ليلية."
        }
    ]
    train_data_source, val_data_source, test_data_source = split_data(data_source_data)
    with open("data_source_train.jsonl", "w", encoding="utf-8") as f:
        for entry in train_data_source:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    with open("data_source_val.jsonl", "w", encoding="utf-8") as f:
        for entry in val_data_source:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    with open("data_source_test.jsonl", "w", encoding="utf-8") as f:
        for entry in test_data_source:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Moves Data
    moves_data = [
        {
            "move": "تحسين السياسة (Policy Optimization)",
            "type": "خوارزمية",
            "objective": "تعديل استراتيجية الوكيل لزيادة المكافأة المتوقعة."
        },
        {
            "move": "تقريب الدالة (Function Approximation)",
            "type": "تقنية",
            "objective": "استخدام الشبكات العصبية لتمثيل دوال القيمة أو السياسات في البيئات المعقدة."
        },
        {
            "move": "التجريد الزمني (Temporal Abstraction)",
            "type": "تقنية",
            "objective": "إنشاء \"حركات كبرى\" (Macro-Actions) تمتد لفترات زمنية طويلة."
        },
        {
            "move": "توليد الأهداف الفرعية (Subgoal Generation)",
            "type": "تقنية",
            "objective": "تقسيم مهمة معقدة إلى أهداف وسيطة أبسط وأكثر قابلية للإدارة."
        },
        {
            "move": "إعادة تشغيل الخبرة (Experience Replay)",
            "type": "تقنية",
            "objective": "تخزين التجارب وإعادة استخدامها لتدريب أكثر كفاءة واستقرارًا."
        },
        {
            "move": "حساب التدخل (Do-calculus)",
            "type": "أداة رياضية",
            "objective": "حساب التأثير السببي لتدخل معين في النظام."
        }
    ]
    train_moves, val_moves, test_moves = split_data(moves_data)
    with open("moves_train.jsonl", "w", encoding="utf-8") as f:
        for entry in train_moves:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    with open("moves_val.jsonl", "w", encoding="utf-8") as f:
        for entry in val_moves:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    with open("moves_test.jsonl", "w", encoding="utf-8") as f:
        for entry in test_moves:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Relationships Data
    relationships_data = [
        {
            "element_a": "عدم كفاءة العينة (RL)",
            "element_b": "التعلم الفوقي / التحويلي",
            "relationship": "B هو حل محتمل لـ A، حيث يهدف إلى تسريع التعلم من بيانات أقل."
        },
        {
            "element_a": "نظرية التحسين (Math)",
            "element_b": "تدريب الشبكات العصبية (RL)",
            "relationship": "A هي الأداة الأساسية لتنفيذ B (مثل استخدام الانحدار التدرجي)."
        },
        {
            "element_a": "الاستدلال العلائقي (Reasoning)",
            "element_b": "التعلم الهرمي (HRL)",
            "relationship": "A يمكن أن يعزز B عبر تمكين فهم أفضل للعلاقات بين الأهداف الفرعية."
        },
        {
            "element_a": "قابلية التوسع (Graphs)",
            "element_b": "الاستدلال العلائقي (Reasoning)",
            "relationship": "تحديات A تؤثر بشكل مباشر على قابلية تطبيق B في المشاكل الكبيرة."
        },
        {
            "element_a": "الأسئلة الفلسفية (Philosophy)",
            "element_b": "جميع التحديات التقنية",
            "relationship": "A توفر الإطار الذي يوجه البحث والأولويات لحل B."
        }
    ]
    train_relationships, val_relationships, test_relationships = split_data(relationships_data)
    with open("relationships_train.jsonl", "w", encoding="utf-8") as f:
        for entry in train_relationships:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    with open("relationships_val.jsonl", "w", encoding="utf-8") as f:
        for entry in val_relationships:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    with open("relationships_test.jsonl", "w", encoding="utf-8") as f:
        for entry in test_relationships:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Challenges Data
    challenges_data = [
        {
            "challenge": "عدم كفاءة العينة",
            "domain": "التعلم المعزز",
            "description": "الحاجة إلى كميات هائلة من البيانات للتعلم بفعالية."
        },
        {
            "challenge": "عدم استقرار التدريب",
            "domain": "التعلم المعزز العميق",
            "description": "حساسية شديدة للمعاملات الفائقة وصعوبة في إعادة إنتاج النتائج."
        },
        {
            "challenge": "التعميم الضعيف",
            "domain": "التعلم المعزز / التمثيل",
            "description": "فشل النموذج في التكيف مع بيئات جديدة تختلف قليلاً عن بيئة التدريب."
        },
        {
            "challenge": "تحديد الهيكل الهرمي",
            "domain": "التعلم المعزز الهرمي",
            "description": "صعوبة تحديد المستويات الصحيحة من التجريد والمهام الفرعية."
        },
        {
            "challenge": "قابلية التفسير",
            "domain": "تعلم التمثيل",
            "description": "صعوبة فهم كيف ولماذا اتخذ النموذج قرارًا معينًا."
        },
        {
            "challenge": "الحد الأدنى المحلي والنقاط السرجية",
            "domain": "نظرية التحسين",
            "description": "إمكانية أن تتعثر خوارزميات التحسين في حلول دون المستوى الأمثل."
        },
        {
            "challenge": "النقل السلبي (Negative Transfer)",
            "domain": "التعلم التحويلي",
            "description": "تدهور الأداء عند نقل المعرفة من مجال مصدر غير مناسب."
        },
        {
            "challenge": "اكتشاف العلاقات السببية",
            "domain": "الاستدلال السببي",
            "description": "صعوبة التمييز بين الارتباط والسببية من البيانات الرصدية وحدها."
        }
    ]
    train_challenges, val_challenges, test_challenges = split_data(challenges_data)
    with open("challenges_train.jsonl", "w", encoding="utf-8") as f:
        for entry in train_challenges:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    with open("challenges_val.jsonl", "w", encoding="utf-8") as f:
        for entry in val_challenges:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    with open("challenges_test.jsonl", "w", encoding="utf-8") as f:
        for entry in test_challenges:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Strategies Data
    strategies_data = [
        {
            "strategy": "استراتيجية \"تعلم لتتعلم\"",
            "core_principle": "استخدام التعلم الفوقي للتكيف السريع مع المهام الجديدة.",
            "targeted_challenges": ["عدم كفاءة العينة", "التعميم الضعيف بين المهام"]
        },
        {
            "strategy": "استراتيجية \"قف على أكتاف العمالقة\"",
            "core_principle": "استخدام التعلم التحويلي للاستفادة من النماذج المدربة مسبقًا.",
            "targeted_challenges": ["عدم كفاءة العينة", "الحاجة إلى بيانات ضخمة"]
        },
        {
            "strategy": "استراتيجية \"فرّق تَسُد\"",
            "core_principle": "استخدام التعلم الهرمي لتقسيم المشاكل المعقدة.",
            "targeted_challenges": ["الأبعاد العالية", "تخطيط طويل المدى", "تصميم المكافآت المتفرقة"]
        },
        {
            "strategy": "استراتيجية \"ابحث عن السبب\"",
            "core_principle": "دمج الاستدلال السببي لفهم أعمق للنظام.",
            "targeted_challenges": ["التعميم خارج نطاق التدريب", "قابلية التفسير", "القرارات غير العادلة"]
        },
        {
            "strategy": "استراتيجية \"ابنِ علاقات\"",
            "core_principle": "استخدام تعلم التمثيل العلائقي لفهم الروابط بين الكيانات.",
            "targeted_challenges": ["البيئات المعقدة ذات الكيانات المتعددة", "التعميم الهيكلي"]
        }
    ]
    train_strategies, val_strategies, test_strategies = split_data(strategies_data)
    with open("strategies_train.jsonl", "w", encoding="utf-8") as f:
        for entry in train_strategies:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    with open("strategies_val.jsonl", "w", encoding="utf-8") as f:
        for entry in val_strategies:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    with open("strategies_test.jsonl", "w", encoding="utf-8") as f:
        for entry in test_strategies:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

if __name__ == '__main__':
    prepare_learning_data()
    prepare_puzzles_data()
    prepare_data_source_data()
    prepare_moves_data()
    prepare_relationships_data()
    prepare_challenges_data()
    prepare_strategies_data()

    prepare_learning_data_csv()
    prepare_puzzles_data_csv()
    prepare_data_source_data_csv()
    prepare_moves_data_csv()
    prepare_relationships_data_csv()
    prepare_challenges_data_csv()
    prepare_strategies_data_csv()

    prepare_learning_data_txt()
    prepare_puzzles_data_txt()
    prepare_data_source_data_txt()
    prepare_moves_data_txt()
    prepare_relationships_data_txt()
    prepare_challenges_data_txt()
    prepare_strategies_data_txt()

    prepare_split_data()


