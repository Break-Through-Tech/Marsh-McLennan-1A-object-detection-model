<html>
<body>
Each NEISS narrative gets one gold object: the consumer product most directly involved in causing the injury. Labels are used to score the object-extraction model, so two labelers reading the same narrative should land on the same answer.</p><p>For every case, record:</p><ul>
<li><strong>gold_object</strong> — a short plain-English name (e.g. "skateboard", "earring")</li>
<li><strong>object_code</strong> — the matching NEISS product code from the <code spellcheck="false" autocorrect="off" autocapitalize="off">NEISS_FMT</code> sheet</li>
<li><strong>object_code_description</strong> — the NEISS name for that code</li>
<li><strong>mention_type</strong> — <code spellcheck="false" autocorrect="off" autocapitalize="off">explicit</code>, <code spellcheck="false" autocorrect="off" autocapitalize="off">implicit</code> or <code spellcheck="false" autocorrect="off" autocapitalize="off">redacted</code> (see rules)</li>
<li><strong>notes</strong> — one line on any judgment call, plus any secondary product and its code</li>
</ul>
<h2 id="labeling-rules">Labeling rules</h2>
<ol>
<li><strong>Label from the narrative, not the coded fields.</strong> Read <code spellcheck="false" autocorrect="off" autocapitalize="off">Narrative_1</code> first and decide on your own. Compare against <code spellcheck="false" autocorrect="off" autocapitalize="off">Product_1</code> only afterward, and note any disagreement.</li>
<li><strong>Pick the product that caused the injury.</strong> If several appear, choose the one whose failure, use or contact produced the harm (the skateboard, not the sidewalk).</li>
<li><strong>Sports and activities count as the object.</strong> "Playing basketball" → basketball (1205), even if no ball is mentioned.</li>
<li><strong>Falls with no named object default to floor (1807).</strong> Use stairs (1842) only when steps are mentioned. Outdoor ground, concrete, grass and dirt are not products.</li>
<li><strong>Body parts, medical devices already implanted and the diagnosis are never the object.</strong> A prosthetic hip dislocating while standing from a chair → chair.</li>
<li><strong>Use the most specific code that fits.</strong> Prefer "earring → jewelry (1616)" over a generic "personal item" code.</li>
<li><strong>Set mention_type honestly.</strong> <code spellcheck="false" autocorrect="off" autocapitalize="off">explicit</code> = object named; <code spellcheck="false" autocorrect="off" autocapitalize="off">implicit</code> = inferred (e.g. floor in a fall); <code spellcheck="false" autocorrect="off" autocapitalize="off">redacted</code> = object hidden behind <code spellcheck="false" autocorrect="off" autocapitalize="off">***</code>.</li>
<li><strong>One secondary object at most</strong>, and only if it materially contributed (the floor a player hit after a basketball fall).</li>
<li><strong>When unsure, label your best guess and flag it</strong> in notes. Don't skip the case; a second labeler resolves flags.</li>
</ol><html>
<body>
Each NEISS narrative gets one gold object: the consumer product most directly involved in causing the injury. Labels are used to score the object-extraction model, so two labelers reading the same narrative should land on the same answer.</p><p>For every case, record:</p><ul>
<li><strong>gold_object</strong> — a short plain-English name (e.g. "skateboard", "earring")</li>
<li><strong>object_code</strong> — the matching NEISS product code from the <code spellcheck="false" autocorrect="off" autocapitalize="off">NEISS_FMT</code> sheet</li>
<li><strong>object_code_description</strong> — the NEISS name for that code</li>
<li><strong>mention_type</strong> — <code spellcheck="false" autocorrect="off" autocapitalize="off">explicit</code>, <code spellcheck="false" autocorrect="off" autocapitalize="off">implicit</code> or <code spellcheck="false" autocorrect="off" autocapitalize="off">redacted</code> (see rules)</li>
<li><strong>notes</strong> — one line on any judgment call, plus any secondary product and its code</li>
</ul>
<h2 id="labeling-rules">Labeling rules</h2>
<ol>
<li><strong>Label from the narrative, not the coded fields.</strong> Read <code spellcheck="false" autocorrect="off" autocapitalize="off">Narrative_1</code> first and decide on your own. Compare against <code spellcheck="false" autocorrect="off" autocapitalize="off">Product_1</code> only afterward, and note any disagreement.</li>
<li><strong>Pick the product that caused the injury.</strong> If several appear, choose the one whose failure, use or contact produced the harm (the skateboard, not the sidewalk).</li>
<li><strong>Sports and activities count as the object.</strong> "Playing basketball" → basketball (1205), even if no ball is mentioned.</li>
<li><strong>Falls with no named object default to floor (1807).</strong> Use stairs (1842) only when steps are mentioned. Outdoor ground, concrete, grass and dirt are not products.</li>
<li><strong>Body parts, medical devices already implanted and the diagnosis are never the object.</strong> A prosthetic hip dislocating while standing from a chair → chair.</li>
<li><strong>Use the most specific code that fits.</strong> Prefer "earring → jewelry (1616)" over a generic "personal item" code.</li>
<li><strong>Set mention_type honestly.</strong> <code spellcheck="false" autocorrect="off" autocapitalize="off">explicit</code> = object named; <code spellcheck="false" autocorrect="off" autocapitalize="off">implicit</code> = inferred (e.g. floor in a fall); <code spellcheck="false" autocorrect="off" autocapitalize="off">redacted</code> = object hidden behind <code spellcheck="false" autocorrect="off" autocapitalize="off">***</code>.</li>
<li><strong>One secondary object at most</strong>, and only if it materially contributed (the floor a player hit after a basketball fall).</li>
<li><strong>When unsure, label your best guess and flag it</strong> in notes. Don't skip the case; a second labeler resolves flags.</li>
</ol><html>
<body>
Each NEISS narrative gets one gold object: the consumer product most directly involved in causing the injury. Labels are used to score the object-extraction model, so two labelers reading the same narrative should land on the same answer.</p><p>For every case, record:</p><ul>
<li><strong>gold_object</strong> — a short plain-English name (e.g. "skateboard", "earring")</li>
<li><strong>object_code</strong> — the matching NEISS product code from the <code spellcheck="false" autocorrect="off" autocapitalize="off">NEISS_FMT</code> sheet</li>
<li><strong>object_code_description</strong> — the NEISS name for that code</li>
<li><strong>mention_type</strong> — <code spellcheck="false" autocorrect="off" autocapitalize="off">explicit</code>, <code spellcheck="false" autocorrect="off" autocapitalize="off">implicit</code> or <code spellcheck="false" autocorrect="off" autocapitalize="off">redacted</code> (see rules)</li>
<li><strong>notes</strong> — one line on any judgment call, plus any secondary product and its code</li>
</ul>
<h2 id="labeling-rules">Labeling rules</h2>
<ol>
<li><strong>Label from the narrative, not the coded fields.</strong> Read <code spellcheck="false" autocorrect="off" autocapitalize="off">Narrative_1</code> first and decide on your own. Compare against <code spellcheck="false" autocorrect="off" autocapitalize="off">Product_1</code> only afterward, and note any disagreement.</li>
<li><strong>Pick the product that caused the injury.</strong> If several appear, choose the one whose failure, use or contact produced the harm (the skateboard, not the sidewalk).</li>
<li><strong>Sports and activities count as the object.</strong> "Playing basketball" → basketball (1205), even if no ball is mentioned.</li>
<li><strong>Falls with no named object default to floor (1807).</strong> Use stairs (1842) only when steps are mentioned. Outdoor ground, concrete, grass and dirt are not products.</li>
<li><strong>Body parts, medical devices already implanted and the diagnosis are never the object.</strong> A prosthetic hip dislocating while standing from a chair → chair.</li>
<li><strong>Use the most specific code that fits.</strong> Prefer "earring → jewelry (1616)" over a generic "personal item" code.</li>
<li><strong>Set mention_type honestly.</strong> <code spellcheck="false" autocorrect="off" autocapitalize="off">explicit</code> = object named; <code spellcheck="false" autocorrect="off" autocapitalize="off">implicit</code> = inferred (e.g. floor in a fall); <code spellcheck="false" autocorrect="off" autocapitalize="off">redacted</code> = object hidden behind <code spellcheck="false" autocorrect="off" autocapitalize="off">***</code>.</li>
<li><strong>One secondary object at most</strong>, and only if it materially contributed (the floor a player hit after a basketball fall).</li>
<li><strong>When unsure, label your best guess and flag it</strong> in notes. Don't skip the case; a second labeler resolves flags.</li>
</ol>