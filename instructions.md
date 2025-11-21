Dokumentacja Projektu: AI Sports Cam
Ten dokument stanowi centralne źródło informacji ("single source of truth") dla projektu AI Sports Cam. Wszystkie generowane rozwiązania muszą być spójne z poniższymi wytycznymi.

1. Misja i Cel Główny Projektu
Tworzymy aplikację mobilną, która fundamentalnie poprawia doświadczenie oglądania amatorskich meczów sportowych. Aplikacja wykorzystuje AI do automatycznego śledzenia akcji na boisku i nagrywania dynamicznego, profesjonalnie wyglądającego wideo, eliminując potrzebę posiadania drogiego sprzętu lub manualnej obsługi kamery.

Produkt końcowy: Aplikacja mobilna (Flutter) na platformy Android i iOS, która w czasie rzeczywistym przetwarza obraz z kamery, kadruje go na głównym obiekcie (piłce) i zapisuje finalne nagranie.

2. Kluczowe Funkcjonalności
Automatyczne Śledzenie AI: Aplikacja musi w czasie rzeczywistym identyfikować i śledzić piłkę na boisku.

Logika "Cyfrowego Kamerzysty": System musi inteligentnie i płynnie kadrować obraz, utrzymując akcję w centrum. Ruchy "kamery" muszą być wygładzone, aby uniknąć gwałtownych skoków.

Nagrywanie na Urządzeniu: Aplikacja musi umożliwiać nagrywanie przetworzonego strumienia wideo i zapisywanie go w pamięci lokalnej telefonu.

Backend i Chmura (Wersja Rozszerzona): System powinien docelowo wspierać konta użytkowników i umożliwiać wysyłanie nagrań do chmury (Google Cloud Storage) w celu archiwizacji i łatwego udostępniania.

3. Architektura Systemu (Full-Stack)
Projekt składa się z trzech głównych, współpracujących ze sobą komponentów:

Aplikacja Mobilna (Flutter):

Odpowiedzialna za interfejs użytkownika (UI/UX).

Zarządza dostępem do kamery i wyświetlaniem podglądu.

Inicjuje proces przetwarzania i zarządza zapisem finalnego wideo.

Moduł Przetwarzania AI (Kod Natywny - Kotlin/Swift):

Działa bezpośrednio na urządzeniu.

Odbiera surowe klatki wideo z kamery.

Uruchamia zoptymalizowany model AI (TensorFlow Lite) w celu detekcji obiektów.

Zwraca współrzędne wykrytych obiektów do warstwy Fluttera.

Backend (Python/FastAPI):

Odpowiedzialny za zarządzanie kontami użytkowników (rejestracja, logowanie).

Stanowi pośrednika w komunikacji z usługami chmurowymi (np. wysyłanie plików do Google Cloud Storage).

4. Stos Technologiczny
Mobile (Front-end): Flutter, Dart

AI / ML (Przetwarzanie): Python (do prototypowania), OpenCV, TensorFlow Lite, model YOLOv8n

Backend: Python, FastAPI

Baza Danych: PostgreSQL (lub SQLite na etapie deweloperskim)

Chmura / DevOps: Google Cloud Platform (Cloud Storage), Docker, Git

5. Fazy Implementacji
Projekt będzie realizowany w następujących, logicznych krokach:

Faza 1: Prototyp na PC: Stworzenie działającego prototypu w Pythonie, który udowodni, że kluczowa logika AI działa zgodnie z założeniami.

Faza 2: Aplikacja Mobilna: Budowa aplikacji we Flutterze z priorytetem na platformę Android, implementacja interfejsu i integracja z modułem AI przez Platform Channels.

Faza 3: Backend i Chmura: Rozbudowa systemu o funkcje sieciowe, w tym konta użytkowników i przechowywanie wideo w chmurze.