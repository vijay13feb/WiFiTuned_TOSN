"""
Split a meeting recording into fixed-length chunks, transcribe each chunk with
Google Speech Recognition, then classify the intent of each transcribed sentence
with a HuggingFace text-classification pipeline.

Requires: pip install SpeechRecognition pydub pandas transformers torch
Also requires ffmpeg on PATH (used by pydub to decode mp3).
"""

import argparse
import os
import shutil

import pandas as pd
import speech_recognition as sr
from pydub import AudioSegment
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, TextClassificationPipeline


def split_audio(input_file, chunks_dir, chunk_length_s):
    if os.path.isdir(chunks_dir):
        shutil.rmtree(chunks_dir)
    os.makedirs(chunks_dir)

    audio = AudioSegment.from_file(input_file)
    total_duration_ms = len(audio)
    chunk_length_ms = chunk_length_s * 1000

    for start_ms in range(0, total_duration_ms, chunk_length_ms):
        chunk = audio[start_ms:start_ms + chunk_length_ms]
        chunk_path = os.path.join(chunks_dir, f"chunk_{start_ms // 1000}.wav")
        chunk.export(chunk_path, format="wav")

    print(f"Split '{input_file}' into {chunk_length_s}-second chunks under '{chunks_dir}'.")


def transcribe_chunks(chunks_dir):
    recognizer = sr.Recognizer()
    sentences = []

    for chunk_name in sorted(os.listdir(chunks_dir)):
        chunk_path = os.path.join(chunks_dir, chunk_name)
        with sr.AudioFile(chunk_path) as source:
            audio_data = recognizer.record(source)

        try:
            text = recognizer.recognize_google(audio_data)
            print(text)
            sentences.append(text)
        except sr.UnknownValueError:
            print(f"Could not understand audio in {chunk_name}")
        except sr.RequestError as e:
            print(f"Speech recognition request failed for {chunk_name}: {e}")

    return sentences


def classify_intents(sentences, model_name):
    tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name)
    text_classifier = TextClassificationPipeline(tokenizer=tokenizer, model=model, return_all_scores=True)

    intents = []
    for sentence in sentences:
        preds = text_classifier(sentence)[0]
        best = max(preds, key=lambda x: x['score'])
        intents.append(best['label'])
        print(f"'{sentence}' -> {best['label']} ({best['score']:.3f})")

    return intents


def parse_args():
    parser = argparse.ArgumentParser(description="Transcribe a meeting recording and classify sentence intents.")
    parser.add_argument('input_audio', help="Path to the source meeting audio file (e.g. .mp3)")
    parser.add_argument('--chunks-dir', default='audio_chunks', help="Directory to store the split audio chunks")
    parser.add_argument('--output-dir', default='intent_output', help="Directory to save the resulting CSV files")
    parser.add_argument('--chunk-length', type=int, default=10, help="Chunk length in seconds")
    parser.add_argument('--model', default='bespin-global/klue-roberta-small-3i4k-intent-classification',
                         help="HuggingFace model used for intent classification")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    split_audio(args.input_audio, args.chunks_dir, args.chunk_length)
    sentences = transcribe_chunks(args.chunks_dir)
    intents = classify_intents(sentences, args.model)

    name = os.path.splitext(os.path.basename(args.input_audio))[0]
    pd.DataFrame(sentences, columns=['sentence']).to_csv(
        os.path.join(args.output_dir, f'sentences_{name}.csv'))
    pd.DataFrame(intents, columns=['intent']).to_csv(
        os.path.join(args.output_dir, f'intent_{name}.csv'))

    print(f"Saved transcripts and intents for '{name}' to '{args.output_dir}'.")
