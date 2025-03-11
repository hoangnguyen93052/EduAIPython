import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, Dropout
from keras.optimizers import Adam
from music21 import converter, instrument, note, chord, stream
import os
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import midi2audio

class MusicGenerator:

    def __init__(self, notes, seq_length=100):
        self.notes = notes
        self.seq_length = seq_length
        self.unique_notes = sorted(set(notes))
        self.n_notes = len(self.unique_notes)
        self.note_to_int = {note: number for number, note in enumerate(self.unique_notes)}
        self.int_to_note = {number: note for number, note in enumerate(self.unique_notes)}
    
    def prepare_sequences(self):
        sequences = []
        targets = []
        for i in range(0, len(self.notes) - self.seq_length):
            seq_in = self.notes[i:i + self.seq_length]
            seq_out = self.notes[i + self.seq_length]
            sequences.append([self.note_to_int[note] for note in seq_in])
            targets.append(self.note_to_int[seq_out])
        X = np.reshape(sequences, (len(sequences), self.seq_length, 1))
        X = X / float(self.n_notes)
        y = np.array(targets)
        return X, y

    def build_model(self):
        model = Sequential()
        model.add(LSTM(256, input_shape=(self.seq_length, 1), return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(256))
        model.add(Dropout(0.3))
        model.add(Dense(self.n_notes))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001))
        return model

    def train_model(self, X, y, epochs=100, batch_size=64):
        y = keras.utils.to_categorical(y, num_classes=self.n_notes)
        self.model = self.build_model()
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size)

    def generate_music(self, start_note, num_notes=500):
        int_start = self.note_to_int[start_note]
        pattern = [int_start]
        output_notes = []

        for _ in range(num_notes):
            prediction_input = np.reshape(pattern, (1, len(pattern), 1))
            prediction_input = prediction_input / float(self.n_notes)
            prediction = self.model.predict(prediction_input, verbose=0)
            index = np.argmax(prediction)
            result = self.int_to_note[index]
            output_notes.append(result)
            pattern.append(index)
            pattern = pattern[1:]

        return output_notes

    def create_midi(self, output_notes, file_name='output_music.mid'):
        output_notes_stream = stream.Stream()
        for note in output_notes:
            if '.' in note or note.isdigit():
                chord_notes = note.split('.')
                notes = [note(int(n)) for n in chord_notes]
                new_chord = chord.Chord(notes)
                output_notes_stream.append(new_chord)
            else:
                new_note = note.Note(note)
                output_notes_stream.append(new_note)
        output_notes_stream.write('midi', fp=file_name)

def load_notes_from_midi_files(midi_folder):
    notes = []
    for file in glob.glob(os.path.join(midi_folder, "*.mid")):
        midi = converter.parse(file)
        notes_to_parse = None
        try:
            notes_to_parse = midi.flat.notes
        except Exception as e:
            print(f"Error parsing {file}: {e}")

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.notes))
    return notes

def main():
    midi_folder = 'path_to_midi_files'  # Update with your MIDI files folder
    notes = load_notes_from_midi_files(midi_folder)

    music_gen = MusicGenerator(notes)
    X, y = music_gen.prepare_sequences()
    
    music_gen.train_model(X, y, epochs=50, batch_size=64)

    start_note = notes[np.random.randint(0, len(notes) - 1)]
    generated_notes = music_gen.generate_music(start_note, num_notes=500)
    
    music_gen.create_midi(generated_notes, file_name='ai_generated_music.mid')
    print("Music generation completed and saved as 'ai_generated_music.mid'.")

if __name__ == '__main__':
    main()