
  Take 1,500 dashcam clips (half contain a crash or near-crash, half are normal driving), automatically organize them into a map where similar driving situations sit near each other, have an AI name each neighborhood ("Rear-end collisions in rain", "Quiet highway at night"), and then check whether the map is actually true.
  
  Phase 0 — Set up the workshop bench. (running now) Install everything, download the dataset and all the AI models, and test every tool once before relying on it. The mantra: any error we can hit today is one we can't hit during the workshop. Also: back everything up to GitHub, because your disk array has no redundancy.

  Phase 1 — Unpack the videos. Turn the 1,500 video files into something fast to work with: extract still frames from every clip once and save them (so we never decode video twice), make one little thumbnail per clip for the map's hover popups, and build a single spreadsheet listing every clip with its label, crash time, and weather/lighting/scene tags.

  Phase 2 — Have the AI watch everything. A vision AI watches all 1,500 clips (plus a zoomed-in look at each crash moment) and writes a short structured description of each: what kind of road, what weather, what's happening, did anything dangerous occur. This gives us text for every video — which matters because the naming tool was built for text.

  Phase 3 — Turn videos into numbers. The three "ways of describing a clip with numbers" from the diagram: counting detected objects, the vision AI's raw impression, and the written descriptions converted to numbers. Also cut out the 2-second event windows. By the end, every clip and window is a row of numbers, and we never need the GPU-heavy stuff again — everything after this is fast analysis.

  Phase 4 — Draw the map and find the neighborhoods. Squash those numbers down to a 2-D map where similar clips land near each other, then automatically find the groups — at several zoom levels (big regions like "highway driving", smaller ones like "highway at night, heavy traffic"). Eyeball check: does each group actually look like one kind of driving?

  Phase 5 — Name the neighborhoods. The big language model looks at each group — its example descriptions, what makes it different from average (e.g. "82% night, 60% rain") — and names it using the official crash-type vocabulary. This is the heart of the project, and where all the trap-avoidance from the research pays off.

  Phase 6 — The honesty checks. Run Test A (hide a few crash clips among many normal ones — do they stand out? whole-clip vs 2-second-window) and Test B (do the names hold up? a judge AI checks each name against clips the namer never saw, and we rebuild everything from scratch to see if the names stay stable). This phase produces the actual numbers for the presentation.

  Phase 7 — Make it beautiful. Build the final interactive map: hover any dot for the video thumbnail and description, search it, browse the topic tree, color it by weather or crash label or "how trustworthy is this name." Must work with no internet — this is the thing people will remember.

  Phase 8 — Bonus rounds, only if time allows. The droppable extras: run Nexar's actual collision-prediction AI over held-out clips and color the map by where it fails ("the predictor struggles in this
  neighborhood"); a couple of alternative-method comparisons; handing the maintainers the bug fix we found in their library.

  Phase 9 — Pack the suitcase. Make sure everything works with the network cable unplugged (the workshop may be air-gapped), rehearse reinstalling from local caches, and build the slide deck.


   1,500 dashcam clips
          │
          ▼
  ┌─────────────────────────────────────────────────┐
  │  STEP 1: Turn each video into numbers            │
  │  (three independent ways, to compare them)       │
  │   a) count what's in it (cars, people, signs)    │
  │   b) a vision AI "looks" at the video            │
  │   c) an AI writes a short description,           │
  │      and we use the text                         │
  └─────────────────────────────────────────────────┘
          │   now every clip is a point in space —
          │   similar clips end up close together
          ▼
  ┌─────────────────────────────────────────────────┐
  │  STEP 2: Make a 2-D map and find the            │
  │  neighborhoods (clusters of similar clips)       │
  └─────────────────────────────────────────────────┘
          ▼
  ┌─────────────────────────────────────────────────┐
  │  STEP 3: An AI names each neighborhood,          │
  │  using the official crash-type vocabulary        │
  │  ("Rear-end", "Pedestrian crossing", ...)        │
  └─────────────────────────────────────────────────┘
          ▼
  ┌─────────────────────────────────────────────────┐
  │  STEP 4: Interactive map — hover over any dot    │
  │  to see the video thumbnail and its description  │
  └─────────────────────────────────────────────────┘
          ▼
  ┌─────────────────────────────────────────────────┐
  │  STEP 5: Two honesty checks                      │
  │   TEST A: hide a few crash clips among lots of   │
  │     normal ones — do they stand out on the map?  │
  │   TEST B: are the names trustworthy? (a second   │
  │     AI checks each name against the clips, and   │
  │     we rebuild the map to see if names change)   │
  └─────────────────────────────────────────────────┘

  The two tests, and why they're the interesting part

  Test A — do crashes stand out? Here's the subtlety: a crash clip is 40 seconds long, but the crash itself lasts about 2 seconds. If you summarize the whole clip, the crash gets diluted — 38 seconds of boring driving drowns out 2 seconds of disaster, and the clip looks normal. So we run the test twice: once on whole clips, once on just the 2-second moment around the event. Our bet is it fails on whole clips and works on the 2-second windows — and that gap is the finding, because it proves that how you chop up the data decides whether the rare stuff is even visible. That's the workshop's central claim, demonstrated on a new kind of data.

  Test B — can you trust AI-written names? A name can sound great and be wrong for half the clips in its group. So we check it two ways: does the name use real crash categories (the official US road-safety vocabulary), and does it actually fit the clips it's attached to.