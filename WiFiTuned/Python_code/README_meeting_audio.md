# Meeting Audio Transcription + Intent Classification

Splits a meeting recording into fixed-length chunks, transcribes each chunk
with Google Speech Recognition, then classifies the intent of each
transcribed sentence with a HuggingFace text-classification pipeline.

Unrelated to the CSI activity-recognition pipeline in this folder — see
`README.md` for that one.

## Requirements

```
pip install SpeechRecognition pydub pandas transformers torch
```

`pydub` also needs `ffmpeg` available on your `PATH` to decode mp3 audio.

## How to Use

```
python3 meeting_audio_intent.py <input_audio.mp3> \
    [--chunks-dir audio_chunks] [--output-dir intent_output] \
    [--chunk-length 10] [--model bespin-global/klue-roberta-small-3i4k-intent-classification]
```

- `input_audio`: path to the source recording (any format `pydub`/`ffmpeg` can decode).
- `--chunks-dir`: where the split `.wav` chunks are written (recreated fresh on every run).
- `--output-dir`: where the resulting CSVs are written (created if missing).
- `--chunk-length`: chunk length in seconds fed to the speech recognizer.
- `--model`: HuggingFace model id used for intent classification.

Output, named after the input file (`<input_audio>` = `meeting.mp3` → `meeting`):

- `<output-dir>/sentences_meeting.csv` — one row per successfully transcribed chunk.
- `<output-dir>/intent_meeting.csv` — the predicted intent label for each row above, same order.

## Known limitation

The default `--model` is a **Korean**-language intent classifier
(`klue-roberta-small-3i4k`). If your meeting audio is in English (or any
non-Korean language), its predictions on the transcribed text won't be
meaningful — pass `--model` with an English intent/sentiment classifier
instead.

## Notes

- Chunks with speech the recognizer can't understand, or that hit a request
  error contacting Google's API, are skipped — `sentences.csv` will have fewer
  rows than there are audio chunks in that case.
- Transcription requires an internet connection (`recognize_google` calls a
  remote API); there's no offline fallback here.
