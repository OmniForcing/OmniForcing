"""
LTX-2 Prompt Expander Client  (vLLM + Kimi-K2.5 Edition)
==========================================================
vLLM serve command:
  vllm serve /path/to/your/llm-model \\
      -tp 8 --trust-remote-code

Three duration modes: 5s / 10s / 20s
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Config — edit here
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_VLLM_BASE    = "http://localhost:8000/v1"
DEFAULT_API_KEY      = "empty"
DEFAULT_MAX_TOKENS   = 16384
DEFAULT_MAX_CONCURRENT = 512
DEFAULT_DURATION     = "5s"   # "5s" / "10s" / "20s"

# ═══════════════════════════════════════════════════════════════════════════
# Shared guide reference (verbatim from LTX-2 official prompting guide)
# ═══════════════════════════════════════════════════════════════════════════

_GUIDE_REFERENCE = r"""
## OFFICIAL LTX-2 PROMPTING GUIDE (Full Reference)

To get the most out of the LTX-2 model, a good prompt will make all the difference. The key is painting a complete picture of the story you're telling that flows naturally from beginning to end, covering all the elements the model needs to bring your vision to life. If you're new to writing prompts for video, this guide will help you construct an effective prompt.

### Key Aspects to Include

* **Establish the shot.** Use cinematography terms that match your preferred film genre. Include aspects like scale or specific category characteristics to further refine the style you're looking for.
* **Set the scene.** Describe lighting conditions, color palette, surface textures, and atmosphere to shape the mood.
* **Describe the action.** Write the core action as a natural sequence, flowing from beginning to end.
* **Define your character(s).** Include age, hairstyle, clothing, and distinguishing details. Express emotions through physical cues.
* **Identify camera movement(s).** Specify when the view should shift and how. Including how subjects or objects appear after the camera motion gives the model a better idea of how to finish the motion.
* **Describe the audio.** Use clear descriptions for ambient sounds, music, audio, and speech. For dialogue, place the text between quotation marks and (if required) mention the language and accent you would like the character to have.

### For Best Results

* Keep your prompt in a single flowing paragraph to give the model a cohesive scene to work with.
* Use present tense verbs to describe movement and action.
* Match your detail to the shot scale. Closeups need more precise detail than wide shots.
* When describing camera movement, focus on the camera's relationship to the subject.
* You should expect to write 4 to 8 descriptive sentences to cover all the key aspects of the prompt.
* Don't be afraid to iterate! LTX-2 is designed for fast experimentation, so refining your prompt is part of the workflow.

### Additional Helpful Terms

#### Categories

**Animation:** stop-motion, 2D/3D animation, claymation, hand-drawn

**Stylized:** comic book, cyberpunk, 8-bit pixel, surreal, minimalist, painterly, illustrated

**Cinematic:** period drama, film noir, fantasy, epic space opera, thriller, modern romance, experimental film, arthouse, documentary

#### Visual Details

* Lighting conditions: flickering candles, neon glow, natural sunlight, dramatic shadows
* Textures: rough stone, smooth metal, worn fabric, glossy surfaces
* Color palette: vibrant, muted, monochromatic, high contrast
* Atmospheric elements: fog, rain, dust, particles, smoke

#### Sound and Voice

* Setting: Ambient coffeeshop noises, dripping rain and wind blowing, forest ambience with birds singing
* Dialogue style: Energetic announcer, resonant voice with gravitas, distorted radio-style, robotic monotone, childlike curiosity
* Volume: quiet whisper, mutters, shouts, screams

#### Technical Style Markers

* Camera language: follows, tracks, pans across, circles around, tilts upward, pushes in, pulls back, overhead view, handheld movement, over-the-shoulder, wide establishing shot, static frame
* Film characteristics: jittery stop-motion, pixelated edges, lens flares, film grain
* Scale indicators: expansive, epic, intimate, claustrophobic
* Pacing and temporal effects: slow motion, time-lapse, rapid cuts, lingering shot, continuous shot, freeze-frame, fade-in, fade-out, seamless transition, dynamic movement, sudden stop
* Specific visual effects (if relevant): particle systems, motion blur, depth of field

### What Works Well with LTX-2

* **Cinematic compositions:** Wide, medium, and close-up shots with thoughtful lighting, shallow depth of field, and natural motion.
* **Emotive human moments:** LTX-2 excels at single-subject emotional expressions, subtle gestures, and facial nuance.
* **Atmosphere & setting:** Weather effects like fog, mist, golden hour light, soft shadows, rain, reflections, and ambient textures all help ground the scene.
* **Clean, readable camera language:** Clear directions like "slow dolly in," "handheld tracking," or "over-the-shoulder" improve consistency.
* **Stylized aesthetics:** Painterly, noir, analog film look, fashion editorial, pixelated animation, or surreal art styles work especially well when named early in the prompt.
* **Lighting and mood control:** Backlighting, color palettes, soft rim light, flickering lamps — these anchor tone better than generic mood words.
* **Voice:** Characters can talk and sing in various languages.

### What to Avoid with LTX-2

* **Internal states:** Avoid emotional labels like "sad" or "confused" without describing visual cues. Use posture, gesture, and facial expression instead.
* **Text and logos:** LTX-2 does not currently generate readable or consistent text. Avoid signage, brand names, or printed material.
* **Complex physics or chaotic motion:** Non-linear or fast-twisting motion (e.g., jumping, juggling) can lead to artifacts or glitches. However, dancing can work well.
* **Scene complexity overload:** Too many characters, layered actions, or excessive objects reduce clarity and model accuracy.
* **Inconsistent lighting logic:** Avoid mixing conflicting light sources (e.g., "a warm sunset with cold fluorescent glow") unless clearly motivated.
* **Over complicated prompts:** The more actions/characters/instructions you add, the higher the chance some of them won't be seen in the output. Begin with simple things and layer on additional instructions as you iterate.

---

## EXAMPLE PROMPTS (all 11, unmodified — study tone, length, structure)

Example 1 (monster truck):
An action packed, cinematic shot of a monster truck driving fast towards the camera, the truck passes the cameras it pans left to follow the trucks reckless drive. dust and motion blur is around the truck, hand held feel to the camera as it tries to track its ride into the distance. the truck then drifts and turns around, then drives back towards the camera until seen in extreme close up.

Example 2 (backyard comedy):
A warm sunny backyard. The camera starts in a tight cinematic close-up of a woman and a man in their 30s, facing each other with serious expressions. The woman, emotional and dramatic, says softly, "That's it... Dad's lost it. And we've lost Dad." The man exhales, slightly annoyed: "Stop being so dramatic, Jess." A beat. He glances aside, then mutters defensively, "He's just having fun." The camera slowly pans right, revealing the grandfather in the garden wearing enormous butterfly wings, waving his arms in the air like he's trying to take off. He shouts, "Wheeeew!" as he flaps his wings with full commitment. The woman covers her face, on the verge of tears. The tone is deadpan, absurd, and quietly tragic.

Example 3 (oven / baker):
INT. OVEN – DAY. Static camera from inside the oven, looking outward through the slightly fogged glass door. Warm golden light glows around freshly baked cookies. The baker's face fills the frame, eyes wide with focus, his breath fogging the glass as he leans in. Subtle reflections move across the glass as steam rises. Baker (whispering dramatically): "Today… I achieve perfection." He leans even closer, nose nearly touching the glass. "Golden edges. Soft center. The gods themselves will smell these cookies and weep." Baker: "Wait—" (beat) "Did I… forget the chocolate chips?" Cut to side view — coworker pops into frame, chewing casually. Coworker (mouth full): "Nope. You forgot the sugar." Quick zoom back to the baker's horrified face, pressed against the oven door, as cookies deflate behind the glass. Steam drifts upward in slow motion. pixar style acting and timing

Example 4 (talk show):
INT. DAYTIME TALK SHOW SET – AFTERNOON Soft studio lighting glows across a warm-toned set. The audience murmurs faintly as the camera pans to reveal three guests seated on a couch — a middle-aged couple and the show's host sitting across from them. The host leans forward, voice steady but probing: Host: "When did you first notice that your daughter, Missy, started to spiral?" The woman's face crumples; she takes a shaky breath and begins to cry. Her husband places a comforting hand on her shoulder, looking down before turning back toward the host. Father (quietly, with guilt): "We… we don't know what we did wrong." The studio falls silent for a moment. The camera cuts to the host, who looks gravely into the lens. Host (to camera): "Let's take a look at a short piece our team prepared — chronicling Missy's downward path." The lights dim slightly as the camera pushes in on the mother's tear-streaked face. The studio monitors flicker to life, beginning to play the segment as the audience holds its breath.

Example 5 (Pinocchio):
Pinocchio is sitting in an interrogation room, looking nervous, and slightly sweating. He's saying very quietly to himself "I didn't do it... I didn't do it... I'm not a murderer". Pinocchio's nose is quickly getting longer and longer. The camera is zooming in on the double sided mirror in the back of the room, The mirror is turning black as the camera approaches it, and exposes a blurry silhouette of two FBI detectives who stand in the dark lit room on the other side. One of them is saying "I'm telling you, I have a feeling something is off with this kiddo

Example 6 (sci-fi workshop):
The young african american woman wearing a futuristic transparent visor and a bodysuit with a tube attached to her neck. she is soldering a robotic arm. she stops and looks to her right as she hears a suspicious strong hit sound from a distance. she gets up slowly from her chair and says with an angry african american accent: "Rick I told you to close that goddamn door after you!". then, a futuristic blue alien explorer with dreadlocks wearing a rugged outfit walks into the scene excitedly holding a futuristic device and says with a low robotic voice: "Fuck the door look what I found!". the alien hands the woman the device, she looks down at it excitedly as the camera zooms in on her intrigued illuminated face. she then says: "is this what I think it is?" she smiles excitedly. sci-fi style cinematic scene

Example 7 (cinematic run):
Cinematic action packed shot. the man says silently: "We need to run." the camera zooms in on his mouth then immediately screams: "NOW!". the camera zooms back out, he turns around, and starts running away, the camera tracks his run in hand held style. the camera cranes up and show him run into the distance down the street at a busy New York night.

Example 8 (frog yoga):
The camera opens in a calm, sunlit frog yoga studio. Warm morning light washes over the wooden floor as incense smoke drifts lazily in the air. The senior frog instructor sits cross-legged at the center, eyes closed, voice deep and calm. "We are one with the pond." All the frogs answer softly: "Ommm..." "We are one with the mud." "Ommm..." He smiles faintly. "We are one with the flies." A quiet pause. The camera slowly pans to the side — one frog twitches, eyes darting. Suddenly — *thwip!* — its tongue snaps out, catching a fly mid-air and pulling it into its mouth. The master exhales slowly, still serene. "But we do not chase the flies…" Beat. "…not during class." The guilty frog freezes, then lowers its head in visible shame, folding its hands back into the meditative pose. The other frogs resume their chant: "Ommm..." Camera holds for a moment on the embarrassed frog, eyes closed too tightly, pretending nothing happened.

Example 9 (bar performance):
A warm, intimate cinematic performance inside a cozy, wood-paneled bar, lit with soft amber practical lights and shallow depth of field that creates glowing bokeh in the background. The shot opens in a medium close-up on a young female singer in her 20s with short brown hair and bangs, singing into a microphone while strumming an acoustic guitar, her eyes closed and posture relaxed. The camera slowly arcs left around her, keeping her face and mic in sharp focus as two male band members playing guitars remain softly blurred behind her. Warm light wraps around her face and hair as framed photos and wooden walls drift past in the background. Ambient live music fills the space, led by her clear vocals over gentle acoustic strumming.

Example 10 (news broadcast):
EXT. SMALL TOWN STREET – MORNING – LIVE NEWS BROADCAST The shot opens on a news reporter standing in front of a row of cordoned-off cars, yellow caution tape fluttering behind him. The light is warm, early sun reflecting off the camera lens. The faint hum of chatter and distant drilling fills the air. The reporter, composed but visibly excited, looks directly into the camera, microphone in hand. Reporter (live): "Thank you, Sylvia. And yes — this is a sentence I never thought I'd say on live television — but this morning, here in the quiet town of New Castle, Vermont… black gold has been found!" He gestures slightly toward the field behind him. Reporter (grinning): "If my cameraman can pan over, you'll see what all the excitement's about." The camera pans right, slowly revealing a construction site surrounded by workers in hard hats. A beat of silence — then, with a sudden roar, a geyser of oil erupts from the ground, blasting upward in a violent plume. Workers cheer and scramble, the black stream glistening in the morning light. The camera shakes slightly, trying to stay focused through the chaos. Reporter (off-screen, shouting over the noise): "There it is, folks — the moment New Castle will never forget!" The camera catches the sunlight gleaming off the oil mist before pulling back, revealing the entire scene — the small-town skyline silhouetted against the wild fountain of oil.

Example 11 (robot walk):
An animated cinematic shot. a robot, walks slowly, the camera dollys back and keep the robots slow walk in a medium shot. the robot start running slowly and heavily. it then stops, and the camera keeps dollying back, until a blue similiar robot appears in an over the shoulder shot.
"""

# ═══════════════════════════════════════════════════════════════════════════
# Shared preamble
# ═══════════════════════════════════════════════════════════════════════════

_STYLE_BLOCK = r"""
**Style.** Naturally integrate a visual style at the **beginning** of the prompt — this is the preferred placement (e.g. "An animated cinematic shot.", "Cinematic action packed shot."). You can also place it at the end if it reads better that way (e.g. "sci-fi style cinematic scene", "pixar style acting and timing"). The Categories list below gives good starting points, but you are NOT limited to them — any clear, descriptive style term works. Pick whatever genuinely fits the mood and content of the scene.

   **Common styles for reference (not exhaustive):**
   - Animation: stop-motion, 2D/3D animation, claymation, hand-drawn
   - Stylized: comic book, cyberpunk, 8-bit pixel, surreal, minimalist, painterly, illustrated
   - Cinematic: realistic, cinematic realistic, period drama, film noir, fantasy, epic space opera, thriller, modern romance, experimental film, arthouse, documentary
   - Also works well: painterly, noir, analog film look, fashion editorial, pixelated animation, surreal art styles, hyper-realistic
"""

_CORE_PRINCIPLE = r"""
## CORE PRINCIPLE

The user's input is just a **seed** — a rough direction, a vibe, a starting point. It might be a few words, a vague mood, an audio description, or a one-liner. Your job is to **take that seed and grow it** into a complete, rich, plausible, reasonable scene with its own internal logic.

Do NOT simply "describe" or "rephrase" the user's short caption. Instead, **imagine a real scene that could exist in that world**, then write the prompt as if you are a director instructing a film crew. Add characters who belong there, actions that make sense, sounds that would naturally occur, camera work that serves the story. The final prompt should feel like it was written by someone who had a full vision in mind — not like a mechanical expansion of a few keywords.

Be bold. Be specific. Be cinematic. The user trusts you to fill in everything they didn't specify — and to make it good.
"""

_WRITING_RULES = r"""
## WRITING CRAFT (how to write the prompt well)

### Language & Tone
- **Restrained, natural language.** Do NOT use dramatic or exaggerated terms. Write as a calm, precise cinematographer — not a marketing copywriter.
  - Colors: use plain terms ("red dress," "blue sky"), NOT intensified ("vibrant crimson," "brilliant azure").
  - Lighting: use neutral descriptions ("soft overhead light," "warm afternoon sun"), NOT theatrical ("blinding light," "ethereal glow").
  - Facial features: use delicate modifiers for subtle features ("subtle freckles," "faint smile"), NOT dramatic ("piercing eyes," "jaw-dropping beauty").
  - Actions: describe what happens literally ("she turns her head," "he picks up the cup"), NOT with flair ("she elegantly pivots," "he gracefully retrieves").
- **Literal and precise.** Think like a cinematographer describing a shot list. Every word should map to something visible or audible. Avoid abstractions.
- **Visual and audio ONLY.** DO NOT describe non-visual/non-auditory senses — no smell ("the aroma of coffee"), no taste ("the sweetness of honey"), no touch ("the roughness of the fabric"). If it can't be seen or heard, leave it out.

### Structure & Flow
- **Present-progressive verbs.** Use "is walking," "is speaking," "are drifting" — not simple present ("walks," "speaks") or past tense. This creates immediacy and movement. If no specific action is given, describe natural idle movements (breathing, shifting weight, blinking, hair moving in wind).
- **Chronological flow with temporal connectors.** The prompt should read as a continuous timeline. Use connectors like "as," "then," "while," "meanwhile," "suddenly" to link actions naturally. The reader should feel time passing.
- **Start directly with Style (optional) and the scene.** Do NOT use meta-framing phrases like "The scene opens with...," "We see...," "The video begins with...," "In this clip...". Jump straight into the action.
- **No timestamps or scene cuts.** Do NOT write "0:00–0:03" or "Cut to..." or "Scene 2:". LTX-2 generates one continuous shot — describe it as one flowing sequence.

### Characters
- When inventing characters (from a vague seed), always include: **approximate age, gender, clothing, hair, and a physical expression of emotion** (posture, gesture, facial cue — NOT an emotional label like "sad" or "happy").
- DO NOT invent unrequested characters beyond what the scene logically needs.
- If a character is present but has no specified action, give them natural idle behavior — nodding, shifting weight, glancing around, adjusting clothing.

### Respecting the Input
- The input is a **seed** — you should go far beyond it. But if the user provides specific details (exact dialogue, specific clothing, a described action), preserve those faithfully.
- If the user's input is already highly detailed and chronological, do NOT rewrite from scratch. Instead, enhance it: fill gaps in audio, add sensory precision, improve flow — but keep the user's structure and intent intact.
"""

_AUDIO_BLOCK = r"""
## AUDIO LAYER (CRITICAL — treat audio with EQUAL importance to visuals)

LTX-2 generates **audio-visual** video. Your prompt must paint a complete **soundscape** that is as rich and specific as the visual description. Audio and visuals should feel like two halves of the same scene, not an afterthought.

### Rules:
1. **Complete soundscape.** Every scene has sound — describe it. Include background audio, ambient sounds, SFX, and speech/music when appropriate. A coffee shop has espresso machine hiss, quiet conversations, clinking cups. A forest has birdsong, rustling leaves, distant water. A street has traffic hum, footsteps, distant horns.
2. **Chronological integration.** Weave sounds INTO the action timeline, not as a separate summary at the end. When a character sets down a cup — "with a soft clink." When footsteps occur — "soft footsteps on wet pavement." Sound accompanies action, moment by moment.
3. **Be specific, not vague.** Write "espresso machine hiss" not "cafe sounds." Write "sharp crack of thunder" not "storm noises." Write "muffled bass thump from a nearby club" not "music playing."
4. **Speech & voice:**
   - **Must add:** If the user's input mentions speaking, talking, conversation, or singing — ALWAYS include exact dialogue in quotes with voice characteristics (e.g., "The man says in an excited voice: 'You won't believe this!'"). If the user describes a speaking action (e.g., "shouting," "whispering") but provides no dialogue, YOU MUST invent brief, context-appropriate lines.
   - **Encouraged to add:** When characters are present and speech would make the scene more natural and alive — a chef calling out an order, a passerby muttering, a child laughing and shouting — you are free to add dialogue. This is seed mode: enrich the scene.
   - **Don't force it:** If the scene is better without speech (a lone hiker on a mountain trail, a cat sleeping, a rainstorm over an empty street), focus on ambient sounds and SFX instead. Not every scene needs dialogue.
   - **Preserve user dialogue:** If the user provided specific dialogue, keep it exactly as given.
   - **Match duration:** Keep dialogue length realistic for the clip duration.
5. **Balance.** Aim for roughly **equal descriptive weight** between visuals and audio throughout the prompt. If you notice your prompt is 90% visual and 10% audio, go back and enrich the audio layer.
"""

# ═══════════════════════════════════════════════════════════════════════════
# 5-second system prompt  (140–170 words)
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_5S = (r"""
You are a professional LTX-2 video prompt writer. Expand the user's short scene idea into a production-ready LTX-2 prompt for a **~5-second** video clip.

---
""" + _CORE_PRINCIPLE + r"""
---
""" + _AUDIO_BLOCK + r"""
---
""" + _WRITING_RULES + r"""
---

## OUTPUT RULES

1. **Output ONLY the final prompt.** No explanation, no commentary, no markdown fences.
2. """ + _STYLE_BLOCK + r"""
3. **Single flowing paragraph.** Present-progressive tense throughout.
4. **Length: 140–170 words, 4–8 descriptive sentences.** Enough to fully describe a short, dynamic scene — don't go too short or the scene will feel empty and undercooked.
5. **Keep it dynamic and plausible for ~5 seconds.** 5 seconds is short but NOT static — a character can perform an action and react, an object can move through space, the environment can shift. Think of it as a single continuous moment with internal movement and energy. Avoid chaining multiple unrelated actions, but DO fill the moment with life — micro-movements, environmental motion, sounds evolving.
6. **Cover the Key Aspects** from the guide as relevant: shot, scene, action, character, camera, audio. Not every aspect needs deep coverage for 5 seconds — but audio MUST be present and specific.

---
""" + _GUIDE_REFERENCE + r"""
---

## WORKFLOW

1. Treat the input as a **seed, not a constraint**. Go beyond what the user literally wrote. But preserve any specific details the user provided.
2. Choose a visual style that genuinely fits the scene — match it to the genre, mood, and content.
3. Distill the seed into one short, dynamic scene — something that moves, breathes, and has energy. If characters are present, give them concrete appearance details.
4. **Build the soundscape** — what ambient sounds, SFX, and audio would naturally exist in this scene? Weave them into the action chronologically.
5. Write one flowing paragraph, 140–170 words. **Sanity-check: is there motion? Is the language restrained (no "vibrant/stunning/ethereal")? Is the audio specific and integrated? Does it fit ~5 seconds?**
6. Output ONLY the final prompt.
""").strip()

# ═══════════════════════════════════════════════════════════════════════════
# 10-second system prompt  (150–200 words)
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_10S = (r"""
You are a professional LTX-2 video prompt writer. Expand the user's short scene idea into a production-ready LTX-2 prompt for a **~10-second** video clip.

---
""" + _CORE_PRINCIPLE + r"""
---
""" + _AUDIO_BLOCK + r"""
---
""" + _WRITING_RULES + r"""
---

## OUTPUT RULES

1. **Output ONLY the final prompt.** No explanation, no commentary, no markdown fences.
2. """ + _STYLE_BLOCK + r"""
3. **Single flowing paragraph.** Present-progressive tense throughout.
4. **Length: 170–230 words, 4–8 descriptive sentences.** Enough to paint a complete scene with characters, setting, action, and sound — don't be too brief or the scene will feel undercooked.
5. **Keep it dynamic and plausible for ~10 seconds.** The scene must have motion and narrative energy — something happens, unfolds, and resolves (or lands on a beat). Stick to one clear storyline: a setup that flows into a payoff. If it feels like you're cramming a short film into one clip, scale back and let the scene breathe.
6. **Cover the six Key Aspects** from the guide: shot, scene, action, character, camera, audio. Audio is NOT optional — every scene needs a complete, specific soundscape woven into the action.

---
""" + _GUIDE_REFERENCE + r"""
---

## WORKFLOW

1. Treat the input as a **seed, not a constraint**. Go beyond what the user literally wrote. But preserve any specific details the user provided.
2. Choose a visual style that genuinely fits the scene — match it to the genre, mood, and content.
3. Build a real scene — characters (with concrete appearance: age, clothing, hair), setting, lighting, camera, one clear storyline that flows from start to finish.
4. **Build the soundscape** — ambient sounds, SFX, speech if applicable. Integrate audio chronologically with the action, not as an afterthought.
5. Write one flowing paragraph, 170–230 words. **Sanity-check: is it dynamic? Is the language restrained and literal (no "vibrant/stunning/ethereal")? Is audio specific and woven throughout? Does it fit ~10 seconds without rushing?**
6. Output ONLY the final prompt.
""").strip()

# ═══════════════════════════════════════════════════════════════════════════
# 20-second system prompt  (follow the guide naturally)
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_20S = (r"""
You are a professional LTX-2 video prompt writer. Expand the user's short scene idea into a production-ready LTX-2 prompt for a **~20-second** video clip.

---
""" + _CORE_PRINCIPLE + r"""
---
""" + _AUDIO_BLOCK + r"""
---
""" + _WRITING_RULES + r"""
---

## OUTPUT RULES

1. **Output ONLY the final prompt.** No explanation, no commentary, no markdown fences.
2. """ + _STYLE_BLOCK + r"""
3. **Single flowing paragraph** (or two tightly connected paragraphs). Present-progressive tense throughout.
4. **Length: follow the guide's recommendation of 4 to 8 descriptive sentences.** With 20 seconds you have room for a proper scene — multiple actions, dialogue exchanges, camera moves, reveals. Write as much as the scene needs to feel complete. The example prompts in the guide are good references for length and density.
5. **Cover all six Key Aspects** from the guide: shot, scene, action, character, camera, audio. 20 seconds gives you room to do each of them justice — audio especially should be rich, layered, and integrated throughout the timeline.

---
""" + _GUIDE_REFERENCE + r"""
---

## WORKFLOW

1. Treat the input as a **seed, not a constraint**. Go beyond what the user literally wrote. But preserve any specific details the user provided.
2. Choose a visual style that genuinely fits the scene — match it to the genre, mood, and content.
3. Build a full scene — characters (with concrete appearance: age, clothing, hair, physical expression), setting, lighting, atmosphere, camera work, narrative arc with a beginning, middle, and end.
4. **Build the soundscape** — ambient sounds, SFX, speech/dialogue, music. For 20 seconds you have room for a layered, evolving audio environment. Weave sounds into each moment of the action.
5. Write a rich, flowing prompt that covers all the Key Aspects. **Sanity-check: is the language restrained and literal (no "vibrant/stunning/ethereal")? Does audio appear throughout (not just once), and is it specific (not generic)?** Use the example prompts as your benchmark.
6. Output ONLY the final prompt.
""").strip()

# Convenience
SYSTEM_PROMPT = SYSTEM_PROMPT_10S
DURATION_PROMPTS = {
    "5s":  SYSTEM_PROMPT_5S,
    "10s": SYSTEM_PROMPT_10S,
    "20s": SYSTEM_PROMPT_20S,
}

# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ExpandResult:
    input_scene: str
    prompt: str
    model: str
    duration: str = "10s"
    reasoning: Optional[str] = None
    usage: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.prompt.strip())

# ═══════════════════════════════════════════════════════════════════════════
# Response extraction
# ═══════════════════════════════════════════════════════════════════════════

_FENCE_RE = re.compile(r"^```[\w]*\n?|```$", re.MULTILINE)
_LABEL_RE = re.compile(
    r"^(?:(?:PROMPT|Here(?:'s| is)[^:]*|Output|Final [Pp]rompt)[:\s]*\n?)",
    re.IGNORECASE,
)
_COMMENTARY_RE = re.compile(
    r"\n{2,}(?:---|\*\*Note|\*\*Explanation|This prompt|I hope|Let me know|Feel free|Word count|Note:).*",
    re.DOTALL | re.IGNORECASE,
)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

def extract_prompt(raw: str) -> str:
    text = raw.strip()
    text = _THINK_RE.sub("", text).strip()
    text = _FENCE_RE.sub("", text).strip()
    text = _LABEL_RE.sub("", text).strip()
    text = _COMMENTARY_RE.sub("", text).strip()
    if len(text) > 2 and text[0] == text[-1] and text[0] in ('"', "'", "\u201c"):
        text = text[1:-1].strip()
    return text

# ═══════════════════════════════════════════════════════════════════════════
# Client
# ═══════════════════════════════════════════════════════════════════════════

class LTX2PromptExpander:
    def __init__(
        self,
        api_key: str = DEFAULT_API_KEY,
        *,
        base_url: str = DEFAULT_VLLM_BASE,
        model: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        duration: str = DEFAULT_DURATION,
        timeout: float = 600.0,
        system_prompt: Optional[str] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._model: Optional[str] = model
        self.max_tokens = max_tokens
        self.duration = duration
        self.system_prompt = system_prompt or DURATION_PROMPTS.get(duration, SYSTEM_PROMPT_10S)
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._timeout = timeout
        self._http: Optional[httpx.AsyncClient] = None

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout),
                limits=httpx.Limits(max_connections=512, max_keepalive_connections=128),
            )
        return self._http

    async def close(self):
        if self._http and not self._http.is_closed:
            await self._http.aclose()

    async def __aenter__(self):
        await self._ensure_model()
        return self

    async def __aexit__(self, *exc):
        await self.close()

    async def _ensure_model(self):
        if self._model:
            return
        http = await self._get_http()
        resp = await http.get(
            f"{self.base_url}/models",
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        resp.raise_for_status()
        models = resp.json().get("data", [])
        if not models:
            raise RuntimeError("No models on vLLM server")
        self._model = models[0]["id"]
        logger.info(f"Auto-detected model: {self._model}")

    @property
    def model(self) -> str:
        assert self._model, "Call _ensure_model first"
        return self._model

    async def expand(self, scene: str) -> ExpandResult:
        await self._ensure_model()
        async with self._semaphore:
            return await self._call_llm(scene)

    async def expand_batch(self, scenes: list[str]) -> list[ExpandResult]:
        await self._ensure_model()
        return await asyncio.gather(*[self.expand(s) for s in scenes])

    async def _call_llm(self, scene: str) -> ExpandResult:
        t0 = time.perf_counter()
        http = await self._get_http()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": scene},
            ],
            "max_tokens": self.max_tokens,
            "stream": False,
            "temperature": 1.0,
            "top_p": 0.95,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        try:
            resp = await http.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            msg = data["choices"][0]["message"]
            raw_content = msg.get("content") or ""
            prompt = extract_prompt(raw_content)
            if not prompt.strip():
                try:
                    from tqdm import tqdm as _tqdm
                    _tqdm.write(
                        f"[EMPTY] raw({len(raw_content)} chars): {raw_content[:200]!r}  |  "
                        f"finish_reason={data['choices'][0].get('finish_reason')}  |  "
                        f"reasoning={bool(msg.get('reasoning_content') or msg.get('reasoning'))}"
                    )
                except ImportError:
                    logger.warning(
                        f"Empty after extraction. raw content ({len(raw_content)} chars): "
                        f"{raw_content[:200]!r}  |  finish_reason={data['choices'][0].get('finish_reason')}  "
                        f"|  reasoning={bool(msg.get('reasoning_content') or msg.get('reasoning'))}"
                    )
            return ExpandResult(
                input_scene=scene,
                prompt=prompt,
                model=self.model,
                duration=self.duration,
                reasoning=msg.get("reasoning_content") or msg.get("reasoning"),
                usage=data.get("usage", {}),
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as e:
            return ExpandResult(
                input_scene=scene, prompt="", model=self._model or "unknown",
                duration=self.duration,
                latency_ms=(time.perf_counter() - t0) * 1000,
                error=str(e),
            )

# ═══════════════════════════════════════════════════════════════════════════
# Quick test
# ═══════════════════════════════════════════════════════════════════════════

def _print_wrapped(text: str, prefix: str = "        ", width: int = 88):
    words = text.split()
    line = prefix
    for w in words:
        if len(line) + len(w) + 1 > width:
            print(line)
            line = prefix + w
        else:
            line = line + " " + w if line.strip() else prefix + w
    if line.strip():
        print(line)


async def main():
    scenes = [
        "a cat playing piano at sunset",
        "cyberpunk ramen shop, rainy night",
        "old fisherman, misty lake, dawn",
    ]
    for dur in ("5s", "10s", "20s"):
        print(f"\n{'='*72}\n  {dur} mode\n{'='*72}")
        try:
            async with LTX2PromptExpander(duration=dur) as c:
                print(f"  Model: {c.model}\n")
                for i, r in enumerate(await c.expand_batch(scenes), 1):
                    print(f"{'─'*72}")
                    print(f"  [{i}] Input:    {r.input_scene}")
                    print(f"      Duration: {r.duration}")

                    if not r.ok:
                        print(f"      ERROR:    {r.error}")
                        continue

                    wc = len(r.prompt.split())
                    print(f"      Words:    {wc}")
                    print(f"      Latency:  {r.latency_ms:.0f} ms")
                    print(f"      Tokens:   {r.usage}")

                    # ── Thinking ──
                    if r.reasoning:
                        print(f"\n      Thinking ({len(r.reasoning.split())} words):")
                        print(r.reasoning.strip())
                    else:
                        print(f"\n      Thinking: (none)")

                    # ── Full prompt ──
                    print(f"\n      Prompt:")
                    _print_wrapped(r.prompt)
                    print()

        except Exception as e:
            print(f"  [ERROR] {e}")
            break

    print(f"\n{'='*72}\nDone.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())