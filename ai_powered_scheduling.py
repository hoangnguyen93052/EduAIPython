import datetime
import random
import json
from typing import List, Dict, Tuple, Optional
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from collections import defaultdict

class Event:
    def __init__(self, title: str, start_time: datetime.datetime, end_time: datetime.datetime, participants: List[str]):
        self.title = title
        self.start_time = start_time
        self.end_time = end_time
        self.participants = participants

    def __repr__(self):
        return f"Event({self.title}, {self.start_time}, {self.end_time}, {self.participants})"

class Schedule:
    def __init__(self):
        self.events: List[Event] = []

    def add_event(self, event: Event) -> bool:
        if not self.is_time_slot_available(event.start_time, event.end_time, event.participants):
            return False
        self.events.append(event)
        return True

    def is_time_slot_available(self, start_time: datetime.datetime, end_time: datetime.datetime, participants: List[str]) -> bool:
        for event in self.events:
            if set(event.participants) & set(participants):
                if (start_time < event.end_time and end_time > event.start_time):
                    return False
        return True

    def get_events(self) -> List[Event]:
        return self.events

class Participant:
    def __init__(self, name: str):
        self.name = name

class AI_Scheduler:
    def __init__(self):
        self.schedule = Schedule()
        self.participant_availability: Dict[str, List[Tuple[datetime.datetime, datetime.datetime]]] = defaultdict(list)

    def add_participant_availability(self, participant: str, start: str, end: str):
        start_time = datetime.datetime.fromisoformat(start)
        end_time = datetime.datetime.fromisoformat(end)
        self.participant_availability[participant].append((start_time, end_time))

    def find_optimal_time_slot(self, duration: int, participants: List[str]) -> Optional[Tuple[datetime.datetime, datetime.datetime]]:
        available_slots = []
        for participant in participants:
            for start, end in self.participant_availability[participant]:
                time_slot = (start, end)
                available_slots.append(time_slot)

        available_slots = sorted(available_slots, key=lambda x: x[0])
        merged_slots = self.merge_time_slots(available_slots)

        for start, end in merged_slots:
            if (end - start).seconds >= duration * 60:
                return (start, start + datetime.timedelta(minutes=duration))

        return None

    def merge_time_slots(self, time_slots: List[Tuple[datetime.datetime, datetime.datetime]]) -> List[Tuple[datetime.datetime, datetime.datetime]]:
        merged = []
        for start, end in time_slots:
            if not merged or merged[-1][1] < start:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        return merged

    def schedule_event(self, title: str, duration: int, participants: List[str]) -> bool:
        optimal_time_slot = self.find_optimal_time_slot(duration, participants)
        if optimal_time_slot:
            start, end = optimal_time_slot
            event = Event(title, start, end, participants)
            return self.schedule.add_event(event)
        return False

    def suggest_event_times(self, title: str, duration: int, participants: List[str]) -> List[Tuple[datetime.datetime, datetime.datetime]]:
        optimal_times = []
        for hour in range(24):
            start_time = datetime.datetime.now().replace(hour=hour, minute=0, second=0, microsecond=0)
            if self.schedule.is_time_slot_available(start_time, start_time + datetime.timedelta(minutes=duration), participants):
                optimal_times.append((start_time, start_time + datetime.timedelta(minutes=duration)))
        return optimal_times

    def get_schedule(self) -> List[Event]:
        return self.schedule.get_events()

class MLModel:
    def __init__(self, data: List[Dict]):
        self.data = data
        self.model = DecisionTreeClassifier()

    def train(self):
        X, y = self.prepare_data()
        self.model.fit(X, y)

    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        X = np.array([[item['duration'], item['participants']] for item in self.data])
        y = np.array([item['scheduled'] for item in self.data])
        return X, y

    def predict(self, event: Event) -> int:
        prediction = self.model.predict([[event.start_time.minute, len(event.participants)]])
        return prediction[0]

class EventManager:
    def __init__(self):
        self.scheduler = AI_Scheduler()
        self.user_data = []
        self.ml_model = None

    def collect_data(self, title: str, start_time: str, end_time: str, participants: List[str]):
        event = Event(title, datetime.datetime.fromisoformat(start_time), datetime.datetime.fromisoformat(end_time), participants)
        scheduled = self.scheduler.schedule_event(title, (end_time - start_time).minutes, participants)
        self.user_data.append({
            'duration': (end_time - start_time).seconds // 60,
            'participants': len(participants),
            'scheduled': int(scheduled)
        })

    def train_ml_model(self):
        self.ml_model = MLModel(self.user_data)
        self.ml_model.train()

    def predict_event(self, event: Event) -> bool:
        if self.ml_model is not None:
            prediction = self.ml_model.predict(event)
            if prediction == 1:
                return self.scheduler.schedule_event(event.title, (event.end_time - event.start_time).minutes, event.participants)
        return False

def main():
    event_manager = EventManager()

    # Adding participant availability
    event_manager.scheduler.add_participant_availability("Alice", "2023-10-10T09:00:00", "2023-10-10T12:00:00")
    event_manager.scheduler.add_participant_availability("Bob", "2023-10-10T10:00:00", "2023-10-10T14:00:00")
    event_manager.scheduler.add_participant_availability("Charlie", "2023-10-10T11:00:00", "2023-10-10T15:00:00")

    # Schedule events
    event_manager.collect_data("Team Meeting", datetime.datetime(2023, 10, 10, 9, 30), datetime.datetime(2023, 10, 10, 10, 30), ["Alice", "Bob", "Charlie"])
    event_manager.collect_data("Project Update", datetime.datetime(2023, 10, 10, 13, 00), datetime.datetime(2023, 10, 10, 14, 00), ["Alice", "Charlie"])

    # Train ML model
    event_manager.train_ml_model()

    # Test scheduling
    new_event = Event("New Team Sync", datetime.datetime(2023, 10, 10, 10, 0), datetime.datetime(2023, 10, 10, 11, 0), ["Alice", "Bob"])
    if event_manager.predict_event(new_event):
        print("Event scheduled successfully!")
    else:
        print("Failed to schedule event.")

if __name__ == "__main__":
    main()