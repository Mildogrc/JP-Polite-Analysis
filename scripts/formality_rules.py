FORMALITY_RULES = [

    # Copula
    ["だ", "です"],
    ["だった", "でした"],
    ["じゃない", "じゃありません", "ではありません"],
    ["じゃなかった", "じゃありませんでした", "ではありませんでした"],
    ["だろう", "でしょう"],
    ["かな", "でしょうか"],
    ["か？", "ですか？"],

    # Basic polite verb conversions
    ["行く", "行きます"],
    ["食べる", "食べます"],
    ["来る", "来ます"],
    ["行こう", "行きましょう"],     # volitional (simple)
    ["行っている", "行っています"],
    ["行った", "行きました"],

    # Imperative → polite request chain
    ["行け",
     "行って",
     "行ってください",
     "行っていただけますか",
     "行ってくださいませんか",
     "行っていただけませんでしょうか"
    ],

    # Negative imperative chain
    ["行くな",
     "行かないで",
     "行かないでください",
     "行かないでいただけますか"
    ],

    # Invitation chain
    ["行こう",
     "行きましょう",
     "行ってみませんか"
    ],

    # Sentence-final particles
    ["よ", "ですよ"],
    ["ね", "ですね"],
    ["かい", "ですか", "ますか"],

    # Rough speech removed or softened
    ["ぞ", ""],       # removed in polite speech
    ["ぜ", ""],
    ["ぞ", "よ"],     # soften
    ["ぜ", "よ"],
    ["さ", ""],

    # Desire forms
    ["行きたい", "行きたいです", "行きたく思います"],

    # Negative → polite negative
    ["行かない", "行きません"],
    ["ない", "ません"],

    # Potential → polite potential
    ["行ける", "行けます"],

    # Very common softeners
    ["ちょっと", "少々"],
    ["すごく", "非常に", "大変"],
    ["でも", "しかし", "ですが"],
    ["ほんとに", "本当に", "誠に"],

    # Polite prefixes
    ["ご", "お"],

    # Honorific & humble forms (single-word transformations)
    ["言う", "おっしゃる"],
    ["行く", "いらっしゃる"],
    ["来る", "いらっしゃる"],
    ["いる", "いらっしゃる"],
    ["する", "なさる"],
    ["見る", "ご覧になる"],
    ["行く", "伺う"],
    ["来る", "伺う"],
    ["もらう", "いただく"],
    ["する", "いたす"],

    # “thinking” forms
    ["と思う", "と思います", "と存じます"],

    # Clause connectors
    ["から", "ので"],
    ["って", "という", "と申します"],
    ["けど", "けれども", "しかしながら"],

    # Additional core rules
    ["て", "てください"],
    ["よう", "ましょう"],        # volitional base
    ["ろ", "て", "てください"],   # imperative softened
    ["よ", "ですよ", ],
    ["ね", "ですね"],
]
