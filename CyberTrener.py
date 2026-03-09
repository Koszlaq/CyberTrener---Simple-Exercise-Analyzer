import mediapipe as mp
import numpy as np
import customtkinter as ctk
from PIL import Image
import math
import threading
import time
import cv2
from datetime import datetime
import os

# ==============================================================================
# 1. KONFIGURACJA
# ==============================================================================
STRONA_BOCZNA = "LEFT"
WYBRANE_CWICZENIE = "SQUAT"
CZY_POKAZAC_SZKIELET = True
PROG_JAKOSCI = 0.7


# ==============================================================================
# 2. KLASA KAMERY IP
# ==============================================================================
class KameraIP:
    def __init__(self, zrodlo, nazwa):
        self.zrodlo = zrodlo
        self.nazwa = nazwa
        self.klatka = None
        self.status = "Laczenie..."
        self.czy_dziala = False
        self.watek = None
        self.lock = threading.Lock()

    def start(self):
        if self.czy_dziala: return
        self.czy_dziala = True
        self.watek = threading.Thread(target=self._update, daemon=True)
        self.watek.start()

    def _update(self):
        print(f"[{self.nazwa}] Laczenie z: {self.zrodlo}")
        cap = cv2.VideoCapture(self.zrodlo)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            self.status = "BLAD POLACZENIA"
            self.czy_dziala = False
            return

        print(f"[{self.nazwa}] Polaczono!")
        self.status = "OK"

        while self.czy_dziala:
            ret, frame = cap.read()
            if ret:
                with self.lock:
                    self.klatka = frame
                    self.status = "OK"
            else:
                self.status = "BRAK SYGNALU"
                time.sleep(0.1)
                cap.release()
                cap = cv2.VideoCapture(self.zrodlo)
        cap.release()

    def pobierz(self):
        with self.lock:
            return self.klatka, self.status

    def stop(self):
        print(f"[{self.nazwa}] Zatrzymywanie kamery...")
        self.czy_dziala = False

        if self.watek and self.watek.is_alive():
            self.watek.join(timeout=0.5)
            if self.watek.is_alive():
                print(f"[{self.nazwa}] Wątek nadal działa, ale będzie zakończony przez daemon")

        print(f"[{self.nazwa}] Kamera zatrzymana")


# ==============================================================================
# 3. KLASA ANALIZATORA (SYNCHRONIZOWANA Z PEŁNYM RAPORTEM)
# ==============================================================================
class AnalizatorTreningu:
    def __init__(self, tryb_analizy, cwiczenie):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.czy_pokazac_szkielet = CZY_POKAZAC_SZKIELET
        self.tryb_analizy = tryb_analizy.upper()
        self.cwiczenie = cwiczenie.upper()

        if "FRONT" in self.tryb_analizy:
            self.tryb = "FRONT"
            self.czy_prawa_strona = "LEFT" not in self.tryb_analizy
        elif "SIDE" in self.tryb_analizy:
            self.tryb = "SIDE"
            self.czy_prawa_strona = "RIGHT" in self.tryb_analizy

        self.powtorzenia = 0
        self.etap = None
        self.komunikat = ""
        self.ostatni_kat_pomocniczy = 0
        self.liczba_klatek = 0
        self.klatki_poprawne = 0
        self.statystyki_bledow = {}
        self.poprzedni_kat = 0

        self.klatki_w_powtorzeniu = 0
        self.bledne_klatki_w_powtorzeniu = 0
        self.ostatni_status_powtorzenia = ""

        self.czy_jest_blad_lokalny = False
        self.czy_jest_blad_zewnetrzny = False

    @staticmethod
    def oblicz_kat_3_punkty(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radiany = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        kat = np.abs(radiany * 180.0 / np.pi)
        if kat > 180.0:
            kat = 360 - kat

        return kat

    @staticmethod
    def rysuj_luk(obraz, srodek, a, c, kolor, promien):
        kat_poczatek = math.atan2(a[1] - srodek[1], a[0] - srodek[0]) * 180 / np.pi
        if kat_poczatek < 0:
            kat_poczatek += 360

        kat_koniec = math.atan2(c[1] - srodek[1], c[0] - srodek[0]) * 180 / np.pi
        if kat_koniec < 0:
            kat_koniec += 360

        if kat_koniec > kat_poczatek:
            if kat_koniec - kat_poczatek > 180:
                cv2.ellipse(obraz, srodek, (promien, promien), 0, kat_koniec, kat_poczatek + 360, kolor, -1)
            else:
                cv2.ellipse(obraz, srodek, (promien, promien), 0, kat_poczatek, kat_koniec, kolor, -1)
        else:
            if kat_poczatek - kat_koniec > 180:
                cv2.ellipse(obraz, srodek, (promien, promien), 0, kat_poczatek, kat_koniec + 360, kolor, -1)
            else:
                cv2.ellipse(obraz, srodek, (promien, promien), 0, kat_koniec, kat_poczatek, kolor, -1)

    def ustaw_blad_zewnetrzny(self, czy_blad):
        self.czy_jest_blad_zewnetrzny = czy_blad

    def sprawdz_i_zalicz_powtorzenie(self):
        if self.klatki_w_powtorzeniu < 5:
            return

        jakosc = 1.0 - (self.bledne_klatki_w_powtorzeniu / self.klatki_w_powtorzeniu)
        if jakosc >= PROG_JAKOSCI:
            self.powtorzenia += 1
            self.ostatni_status_powtorzenia = "ZALICZONE"
        else:
            self.ostatni_status_powtorzenia = "NIEZALICZONE (BLEDY)"

        self.klatki_w_powtorzeniu = 0
        self.bledne_klatki_w_powtorzeniu = 0

    def aktualizuj_statystyki(self, aktualny_kat):
        delta = abs(aktualny_kat - self.poprzedni_kat)
        self.poprzedni_kat = aktualny_kat
        if delta < 0.8:
            return

        self.liczba_klatek += 1
        self.klatki_w_powtorzeniu += 1

        dobre = ["IDEALNIE", "POSTAWA OK", "PELNY WYPROST!", "PEŁNY ZAKRES!", "DOBRZE", "PRZYGOTUJ SIE", "PRZYSIAD", ""]
        if self.komunikat in dobre:
            self.czy_jest_blad_lokalny = False
            self.klatki_poprawne += 1
        else:
            self.czy_jest_blad_lokalny = True
            self.statystyki_bledow[self.komunikat] = self.statystyki_bledow.get(self.komunikat, 0) + 1

        if self.czy_jest_blad_lokalny or self.czy_jest_blad_zewnetrzny:
            self.bledne_klatki_w_powtorzeniu += 1

    def generuj_raport(self):
        katalog_raportow = "raporty_treningowe"
        if not os.path.exists(katalog_raportow):
            os.makedirs(katalog_raportow)

        czas_i_data = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        nazwa_pliku = f"{katalog_raportow}/raport_{self.cwiczenie}_{self.tryb}_{czas_i_data}.txt"

        raport_linie = []
        raport_linie.append("=" * 60)
        raport_linie.append(f"RAPORT SESJI TRENINGOWEJ")
        raport_linie.append("=" * 60)
        raport_linie.append(f"Ćwiczenie: {self.cwiczenie}")
        raport_linie.append(f"Kamera: {self.tryb}")
        raport_linie.append(f"Data i czas: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        raport_linie.append("=" * 60)
        raport_linie.append("")

        if self.liczba_klatek == 0:
            raport_linie.append("UWAGA: Brak zarejestrowanych danych treningowych.")
            raport_linie.append("Możliwe przyczyny:")
            raport_linie.append("  - Kamera nie była podłączona")
            raport_linie.append("  - Brak wykrytej osoby w kadrze")
            raport_linie.append("  - Sesja została zakończona zaraz po uruchomieniu")
            raport_linie.append("  - Te ćwiczenie nie analizuje błędów z bocznej kamery")
            raport_linie.append("")
        else:
            raport_linie.append(f"Powtórzeń zaliczonych: {self.powtorzenia}")
            jakosc = (self.klatki_poprawne / self.liczba_klatek) * 100
            raport_linie.append(f"Technika: {jakosc:.1f}%")

        porady_slownik = {
            "NIE BUJAJ PLECAMI": "Zepnij brzuch i pośladki. Tułów musi być stabilny.",
            "NIE ODCHYLAJ PLECOW!": "Zbyt mocny przeprost lędźwi. Ściśnij pośladki.",
            "WYROWNAJ SZTANGE": "Jedna ręka pracuje szybciej. Wyrównaj tempo.",
            "BARKI KRZYWO": "Ustaw barki w jednej linii.",
            "SZERSZY CHWYT": "Twój chwyt jest zbyt wąski. Rozsuń dłonie.",
            "WEZSZY CHWYT": "Trzymasz sztangę zbyt szeroko. Zsuń dłonie.",
            "LOKCIE WEZIEJ": "Trzymaj łokcie pod sztangą, nie na boki.",
            "LOKCIE WASKI": "Łokcie rozjeżdżają się na boki. Trzymaj je blisko uszu.",
            "SZTANGA NAD GLOWE": "Wypchnij sztangę lekko do tyłu w górze.",
            "UNIES LOKCIE": "Łokcie muszą celować w sufit/wysoko.",
            "LOKCIE WYZEJ": "Łokcie opadają do przodu. Pionowe ramię.",
            "NIZEJ": "Zejdź głębiej (biodra poniżej kolan).",
            "PLECY PIONOWO": "Nie pochylaj się tak mocno do przodu.",
            "KOLANA NA ZEWNATRZ": "Kolana schodzą do środka! Rozpychaj je na zewnątrz."
        }

        if self.statystyki_bledow and self.liczba_klatek > 0:
            raport_linie.append("")
            raport_linie.append("=" * 60)
            raport_linie.append("WYKRYTE PROBLEMY TECHNICZNE:")
            raport_linie.append("=" * 60)

            for blad, ilosc in sorted(self.statystyki_bledow.items(), key=lambda x: x[1], reverse=True):
                proc = (ilosc / self.liczba_klatek) * 100
                if proc > 2.0:
                    raport_linie.append(f"\n• {blad}:")
                    raport_linie.append(f"  - Częstotliwość: {proc:.1f}% czasu")
                    raport_linie.append(f"  - Porada: {porady_slownik.get(blad, 'Popraw to.')}")
        elif self.liczba_klatek > 0:
            raport_linie.append("")
            raport_linie.append("Gratulacje! Nie wykryto błędów technicznych.")

        raport_linie.append("")
        raport_linie.append("=" * 60)
        raport_linie.append("Koniec raportu")
        raport_linie.append("=" * 60)

        try:
            with open(nazwa_pliku, 'w', encoding='utf-8') as plik:
                plik.write('\n'.join(raport_linie))

            print(f"Raport zapisany: {nazwa_pliku}")
            return nazwa_pliku
        except Exception as e:
            print(f"Błąd przy zapisywaniu raportu: {e}")
            return None

    def przetwarzaj_klatke(self, klatka):
        obraz_rgb = cv2.cvtColor(klatka, cv2.COLOR_BGR2RGB)
        h, w, _ = obraz_rgb.shape
        wyniki = self.pose.process(obraz_rgb)
        obraz = cv2.cvtColor(obraz_rgb, cv2.COLOR_RGB2BGR)

        if wyniki.pose_landmarks:
            punkty = wyniki.pose_landmarks.landmark

            def pobierz_px(idx):
                return (int(punkty[idx].x * w), int(punkty[idx].y * h))

            def pobierz_norm(idx):
                return [punkty[idx].x, punkty[idx].y]

            if self.czy_prawa_strona:
                ramie, lokiec, nadgarstek, biodro, kolano, kostka = 12, 14, 16, 24, 26, 28
                widoczne_indeksy = [12, 14, 16, 24, 26, 28]
            else:
                ramie, lokiec, nadgarstek, biodro, kolano, kostka = 11, 13, 15, 23, 25, 27
                widoczne_indeksy = [11, 13, 15, 23, 25, 27]

            kat_do_rysunku = 0
            centrum_luku = lokiec

            if self.cwiczenie == "CURL":
                if self.tryb == "SIDE":
                    pion = [pobierz_norm(biodro)[0], pobierz_norm(biodro)[1] - 0.5]
                    kat_plecow = self.oblicz_kat_3_punkty(pion, pobierz_norm(biodro), pobierz_norm(ramie))
                    centrum_luku = biodro

                    if kat_plecow > 10:
                        self.komunikat, kolor_statusu = "NIE BUJAJ PLECAMI", (0, 0, 255)
                    else:
                        self.komunikat, kolor_statusu = "POSTAWA OK", (0, 255, 0)

                    kat_do_rysunku = kat_plecow

                elif self.tryb == "FRONT":
                    kat_lokcia = self.oblicz_kat_3_punkty(pobierz_norm(ramie), pobierz_norm(lokiec), pobierz_norm(nadgarstek))

                    if self.etap is None:
                        self.etap = "DOL"
                    if kat_lokcia < 40:
                        self.etap = "GORA"
                    if kat_lokcia > 160 and self.etap == "GORA":
                        self.etap = "DOL"
                        self.sprawdz_i_zalicz_powtorzenie()

                    wysokosc_nadgarstkow = abs(punkty[15].y - punkty[16].y)
                    szerokosc_barki = abs(punkty[11].x - punkty[12].x)
                    szerokosc_nadgarstki = abs(punkty[15].x - punkty[16].x)
                    ratio = szerokosc_nadgarstki / szerokosc_barki

                    if wysokosc_nadgarstkow > 0.08:
                        self.komunikat, kolor_statusu = "WYROWNAJ SZTANGE", (0, 0, 255)
                    elif ratio < 1.2:
                        self.komunikat, kolor_statusu = "SZERSZY CHWYT", (0, 0, 255)
                    elif ratio > 2.2:
                        self.komunikat, kolor_statusu = "WEZSZY CHWYT", (0, 0, 255)
                    else:
                        self.komunikat, kolor_statusu = "IDEALNIE", (0, 255, 0)

                    kat_do_rysunku = kat_lokcia

            elif self.cwiczenie == "OHP":
                if self.tryb == "SIDE":
                    pion = [pobierz_norm(biodro)[0], pobierz_norm(biodro)[1] - 0.5]
                    kat_plecow = self.oblicz_kat_3_punkty(pion, pobierz_norm(biodro), pobierz_norm(ramie))
                    centrum_luku = biodro

                    if kat_plecow > 10:
                        self.komunikat, kolor_statusu = "NIE ODCHYLAJ PLECOW!", (0, 0, 255)
                    else:
                        self.komunikat, kolor_statusu = "POSTAWA OK", (0, 255, 0)

                    kat_do_rysunku = kat_plecow

                elif self.tryb == "FRONT":
                    kat_lokcia = self.oblicz_kat_3_punkty(pobierz_norm(ramie), pobierz_norm(lokiec), pobierz_norm(nadgarstek))
                    if kat_lokcia < 70:
                        self.etap = "DOL"
                    if kat_lokcia > 160 and self.etap == "DOL":
                        self.etap = "GORA"
                        self.sprawdz_i_zalicz_powtorzenie()

                    wysokosc_nadgarstkow = abs(punkty[15].y - punkty[16].y)
                    wysokosc_barki = abs(punkty[11].y - punkty[12].y)
                    szerokosc_lokci = abs(punkty[13].x - punkty[14].x)
                    szerokosc_barki = abs(punkty[11].x - punkty[12].x)

                    if wysokosc_nadgarstkow > 0.08:
                        self.komunikat, kolor_statusu = "WYROWNAJ SZTANGE", (0, 0, 255)
                    elif wysokosc_barki > 0.04:
                            self.komunikat, kolor_statusu = "BARKI KRZYWO", (0, 0, 255)
                    elif szerokosc_lokci > szerokosc_barki * 2.5 and kat_lokcia < 90:
                        self.komunikat, kolor_statusu = "LOKCIE WEZIEJ", (0, 0, 255)
                    elif kat_lokcia > 150 and punkty[16].y > punkty[8].y:
                        self.komunikat, kolor_statusu = "SZTANGA NAD GLOWE", (0, 0, 255)
                    else:
                        self.komunikat, kolor_statusu = "IDEALNIE", (0, 255, 0)

                    kat_do_rysunku = kat_lokcia

            elif self.cwiczenie == "TRICEPS":
                if self.tryb == "SIDE":
                    pion = [pobierz_norm(biodro)[0], pobierz_norm(biodro)[1] - 0.5]
                    kat_plecow = self.oblicz_kat_3_punkty(pion, pobierz_norm(biodro), pobierz_norm(ramie))
                    kat_ramienia = self.oblicz_kat_3_punkty(pobierz_norm(biodro), pobierz_norm(ramie), pobierz_norm(lokiec))

                    if kat_plecow > 10:
                        self.komunikat, kolor_statusu = "NIE ODCHYLAJ PLECOW!", (0, 0, 255)
                        kat_do_rysunku = kat_plecow
                        centrum_luku = biodro
                    elif kat_ramienia < 140:
                        self.komunikat, kolor_statusu = "LOKCIE WYZEJ", (0, 0, 255)
                        kat_do_rysunku = kat_ramienia
                        centrum_luku = ramie
                    else:
                        self.komunikat, kolor_statusu = "POSTAWA OK", (0, 255, 0)
                        kat_do_rysunku = kat_plecow
                        centrum_luku = biodro

                elif self.tryb == "FRONT":
                    kat_lokcia = self.oblicz_kat_3_punkty(pobierz_norm(ramie), pobierz_norm(lokiec), pobierz_norm(nadgarstek))
                    if kat_lokcia < 55:
                        self.etap = "DOL"
                    if kat_lokcia > 160 and self.etap == "DOL":
                        self.etap = "GORA"
                        self.sprawdz_i_zalicz_powtorzenie()

                    wysokosc_nadgarstkow = abs(punkty[15].y - punkty[16].y)
                    szerokosc_lokci = abs(punkty[13].x - punkty[14].x)
                    szerokosc_barki = abs(punkty[11].x - punkty[12].x)

                    if wysokosc_nadgarstkow > 0.08:
                        self.komunikat, kolor_statusu = "WYROWNAJ SZTANGE", (0, 0, 255)
                    elif (punkty[13].y > punkty[11].y - 0.05) or (punkty[14].y > punkty[12].y - 0.05):
                        self.komunikat, kolor_statusu = "UNIES LOKCIE", (0, 0, 255)
                    elif szerokosc_lokci > szerokosc_barki * 1.8 and kat_lokcia < 90:
                        self.komunikat, kolor_statusu = "LOKCIE WASKI", (0, 0, 255)
                    else:
                        self.komunikat, kolor_statusu = "IDEALNIE", (0, 255, 0)

                    kat_do_rysunku = kat_lokcia
                    centrum_luku = ramie

            elif self.cwiczenie == "SQUAT":
                if self.tryb == "FRONT":
                    kat_zgiecia_nogi = self.oblicz_kat_3_punkty(pobierz_norm(biodro), pobierz_norm(kolano), pobierz_norm(kostka))
                    szerokosc_kolan = abs(punkty[kolano].x - punkty[kostka].x)
                    szerokosc_kostek = abs(punkty[kolano].x - punkty[kostka].x)

                    if self.etap is None:
                        self.etap = "GORA"
                    if kat_zgiecia_nogi < 80:
                        self.etap = "DOL"
                    if kat_zgiecia_nogi > 150 and self.etap == "DOL":
                        self.etap = "GORA"
                        self.sprawdz_i_zalicz_powtorzenie()

                    if kat_zgiecia_nogi < 165:
                        if szerokosc_kolan < szerokosc_kostek * 0.8:
                            self.komunikat, kolor_statusu = "KOLANA NA ZEWNATRZ", (0, 0, 255)
                        else:
                            self.komunikat, kolor_statusu = "DOBRZE", (0, 255, 0)
                    else:
                        self.komunikat, kolor_statusu = "PRZYGOTUJ SIE", (255, 255, 255)

                    kat_do_rysunku = kat_zgiecia_nogi
                    centrum_luku = kolano

                elif self.tryb == "SIDE":
                    pion = [pobierz_norm(biodro)[0], pobierz_norm(biodro)[1] - 0.5]
                    kat_zgiecia_nogi = self.oblicz_kat_3_punkty(pobierz_norm(biodro), pobierz_norm(kolano), pobierz_norm(kostka))
                    kat_plecow = self.oblicz_kat_3_punkty(pion, pobierz_norm(biodro), pobierz_norm(ramie))
                    biodro_y = punkty[biodro].y
                    kolano_y = punkty[kolano].y

                    if self.etap == "DOL" and kat_zgiecia_nogi < 90:
                        if biodro_y < kolano_y - 0.05:
                            self.komunikat, kolor_statusu = "NIZEJ", (0, 0, 255)
                        elif kat_plecow > 45:
                            self.komunikat, kolor_statusu = "PLECY PIONOWO", (0, 0, 255)
                        else:
                            self.komunikat, kolor_statusu = "DOBRZE", (0, 255, 0)
                    else:
                        self.komunikat, kolor_statusu = "PRZYSIAD", (255, 255, 255)

                    kat_do_rysunku = kat_zgiecia_nogi
                    centrum_luku = kolano

            self.aktualizuj_statystyki(kat_do_rysunku)

            if self.czy_pokazac_szkielet:
                if self.tryb == "SIDE":
                    punkty = [(ramie, lokiec), (lokiec, nadgarstek), (ramie, biodro), (biodro, kolano), (kolano, kostka)]
                    for start, koniec in punkty:
                        cv2.line(obraz, pobierz_px(start), pobierz_px(koniec), (255, 255, 255), 2)
                    for idx in widoczne_indeksy:
                        cv2.circle(obraz, pobierz_px(idx), 5, (245, 117, 16), -1)
                else:
                    for polaczenie in self.mp_pose.POSE_CONNECTIONS:
                        start, koniec = polaczenie
                        if start > 10 and koniec > 10:
                            cv2.line(obraz, pobierz_px(start), pobierz_px(koniec), (255, 255, 255), 2)

                if centrum_luku != 0:
                    nakladka = obraz.copy()
                    if centrum_luku == biodro:
                        pion_rys = (pobierz_px(biodro)[0], pobierz_px(biodro)[1] - 100)
                        self.rysuj_luk(nakladka, pobierz_px(biodro), pion_rys, pobierz_px(ramie), (0, 255, 255), 45)
                    elif centrum_luku == kolano:
                        self.rysuj_luk(nakladka, pobierz_px(kolano), pobierz_px(biodro), pobierz_px(kostka), (0, 255, 255), 45)
                    else:
                        self.rysuj_luk(nakladka, pobierz_px(centrum_luku), pobierz_px(ramie), pobierz_px(nadgarstek), (0, 255, 255), 35)
                    cv2.addWeighted(nakladka, 0.4, obraz, 0.6, 0, obraz)
                    cv2.putText(obraz, str(int(kat_do_rysunku)), (pobierz_px(centrum_luku)[0] - 40, pobierz_px(centrum_luku)[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return obraz


# ==============================================================================
# 4. FRONTEND
# ==============================================================================

class CyberTrenerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("CYBER TRENER v3.0")
        self.geometry("600x450")
        ctk.set_appearance_mode("dark")

        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.cam_front = None
        self.cam_side = None
        self.logic_front = None
        self.logic_side = None
        self.czy_trening_trwa = False
        self.update_job = None

        self.aktualne_cwiczenie = WYBRANE_CWICZENIE
        self.aktualna_strona = STRONA_BOCZNA
        self.pokazuj_szkielet = CZY_POKAZAC_SZKIELET
        self.prog_jakosci = PROG_JAKOSCI

        self._setup_launcher()

    def _setup_launcher(self):
        self.launcher_frame = ctk.CTkFrame(self)
        self.launcher_frame.pack(expand=True, fill="both", padx=20, pady=20)

        ctk.CTkLabel(self.launcher_frame, text="CYBER TRENER SETUP", font=("Segoe UI", 24, "bold")).pack(pady=20)

        ctk.CTkLabel(self.launcher_frame, text="IP Kamery FRONT (np. 192.168.0.9:8080):").pack()
        self.ip_front_entry = ctk.CTkEntry(self.launcher_frame, width=300)
        self.ip_front_entry.insert(0, "192.168.0.3:8080")
        self.ip_front_entry.pack(pady=5)

        ctk.CTkLabel(self.launcher_frame, text="IP Kamery SIDE:").pack()
        self.ip_side_entry = ctk.CTkEntry(self.launcher_frame, width=300)
        self.ip_side_entry.insert(0, "192.168.0.7:8080")
        self.ip_side_entry.pack(pady=5)

        self.start_btn = ctk.CTkButton(self.launcher_frame, text="POŁĄCZ I START", command=self._start_countdown, font=("Segoe UI", 16, "bold"), fg_color="#386641")
        self.start_btn.pack(pady=30)

    def _start_countdown(self):
        ip_f = f"http://{self.ip_front_entry.get()}/video"
        ip_s = f"http://{self.ip_side_entry.get()}/video"

        self.launcher_frame.pack_forget()
        self.count_label = ctk.CTkLabel(self, text="3", font=("Segoe UI", 120, "bold"))
        self.count_label.pack(expand=True)

        def countdown(count):
            if count > 0:
                self.count_label.configure(text=str(count))
                self.after(1000, lambda: countdown(count - 1))
            else:
                self.count_label.configure(text="GO!")
                self.after(500, lambda: self._launch_main_gui(ip_f, ip_s))

        countdown(3)

    def _launch_main_gui(self, ip_f, ip_s):
        self.count_label.pack_forget()
        self.geometry("1500x900")

        self.cam_front = KameraIP(ip_f, "FRONT")
        self.cam_side = KameraIP(ip_s, "SIDE")
        self.cam_front.start()
        self.cam_side.start()

        self.logic_front = AnalizatorTreningu("FRONT", self.aktualne_cwiczenie)
        self.logic_side = AnalizatorTreningu(f"SIDE_{self.aktualna_strona}", self.aktualne_cwiczenie)

        self.czy_trening_trwa = True
        self._setup_ui()
        self.update_frame()

    def _setup_ui(self):
        self.grid_columnconfigure(0, minsize=300)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.sidebar.grid_propagate(False)

        ctk.CTkLabel(self.sidebar, text="KONTROLA", font=("Segoe UI", 24, "bold")).pack(pady=20)

        ctk.CTkLabel(self.sidebar, text="WYBIERZ ĆWICZENIE:", font=("Arial", 11, "bold")).pack(padx=20, anchor="w")
        self.exercise_menu = ctk.CTkOptionMenu(self.sidebar, values=["CURL", "OHP", "TRICEPS", "SQUAT"], command=self.change_exercise)
        self.exercise_menu.set(self.aktualne_cwiczenie)
        self.exercise_menu.pack(pady=(5, 15), padx=20, fill="x")

        ctk.CTkLabel(self.sidebar, text="STRONA KAMERY BOCZNEJ:", font=("Arial", 11, "bold")).pack(padx=20, anchor="w")
        self.side_menu = ctk.CTkOptionMenu(self.sidebar, values=["LEFT", "RIGHT"], command=self.change_side)
        self.side_menu.set(self.aktualna_strona)
        self.side_menu.pack(pady=(5, 15), padx=20, fill="x")

        ctk.CTkLabel(self.sidebar, text="DOKŁADNOŚĆ (PRÓG):", font=("Arial", 11, "bold")).pack(padx=20, anchor="w")
        self.threshold_slider = ctk.CTkSlider(self.sidebar, from_=0.1, to=1.0, number_of_steps=18, command=self.change_threshold)
        self.threshold_slider.set(self.prog_jakosci)
        self.threshold_slider.pack(pady=(5, 0), padx=20, fill="x")
        self.threshold_label = ctk.CTkLabel(self.sidebar, text=f"Wartość: {self.prog_jakosci}", font=("Arial", 10))
        self.threshold_label.pack()

        ctk.CTkLabel(self.sidebar, text="STATUS TECHNIKI:", font=("Segoe UI", 12, "bold")).pack(pady=(20, 5), padx=20, anchor="w")
        self.feedback_label = ctk.CTkLabel(self.sidebar, text="GOTOWY", font=("Segoe UI", 16, "bold"), height=100, corner_radius=8, fg_color="#2b2b2b", wraplength=260)
        self.feedback_label.pack(pady=10, padx=20, fill="x")

        self.reps_frame = ctk.CTkFrame(self.sidebar, fg_color="#1f538d", corner_radius=15)
        self.reps_frame.pack(fill="x", padx=20, pady=10)
        self.reps_val = ctk.CTkLabel(self.reps_frame, text="0", font=("Arial", 64, "bold"))
        self.reps_val.pack()

        self.stop_btn = ctk.CTkButton(self.sidebar, text="STOP I RAPORT", command=self.stop_training, fg_color="#ae2012", hover_color="#78100a", font=("Segoe UI", 14, "bold"))
        self.stop_btn.pack(side="bottom", pady=20, padx=20, fill="x")

        self.skeleton_switch = ctk.CTkSwitch(self.sidebar, text="Pokaż szkielet", command=self.toggle_skeleton)
        if self.pokazuj_szkielet: self.skeleton_switch.select()
        self.skeleton_switch.pack(side="bottom", pady=10, padx=20, anchor="w")

        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self.exercise_title_label = ctk.CTkLabel(self.main_container, text=f"SESJA: {self.aktualne_cwiczenie}", font=("Segoe UI", 28, "bold"), text_color="#3b8ed0")
        self.exercise_title_label.pack(pady=(10, 15))

        self.video_wrapper = ctk.CTkFrame(self.main_container, fg_color="#0a0a0a", corner_radius=15)
        self.video_wrapper.pack(expand=True, fill="both")
        self.video_display = ctk.CTkLabel(self.video_wrapper, text="")
        self.video_display.pack(expand=True, fill="both")

    def update_frame(self):
        if not self.czy_trening_trwa:
            return

        if not self.cam_front or not self.cam_side:
            return

        if not self.logic_front or not self.logic_side:
            return

        try:
            f1, s1 = self.cam_front.pobierz()
            f2, s2 = self.cam_side.pobierz()
            if not self.czy_trening_trwa:
                return

            self.logic_front.ustaw_blad_zewnetrzny(self.logic_side.czy_jest_blad_lokalny)
            img_f = self.logic_front.przetwarzaj_klatke(cv2.resize(f1, (640, 480))) if f1 is not None else np.zeros((480, 640, 3), dtype=np.uint8)
            img_s = self.logic_side.przetwarzaj_klatke(cv2.resize(f2, (640, 480))) if f2 is not None else np.zeros((480, 640, 3), dtype=np.uint8)
            combined = np.hstack((img_f, img_s))

            if not hasattr(self, 'video_wrapper'):
                return
            try:
                if not self.video_wrapper.winfo_exists():
                    return
            except:
                return

            win_w, win_h = self.video_wrapper.winfo_width(), self.video_wrapper.winfo_height()
            if win_w > 100 and win_h > 100:
                oh, ow = combined.shape[:2]
                aspect = ow / oh
                if (win_w / win_h) > aspect:
                    fw, fh = int(win_h * aspect), win_h
                else:
                    fw, fh = win_w, int(win_w / aspect)
                img_tk = ctk.CTkImage(light_image=Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)), size=(fw, fh))
                if not self.czy_trening_trwa:
                    return

                self.video_display.configure(image=img_tk)

            if hasattr(self, 'reps_val') and self.reps_val.winfo_exists():
                self.reps_val.configure(text=str(self.logic_front.powtorzenia))
            if hasattr(self, 'feedback_label') and self.feedback_label.winfo_exists():
                if self.logic_front.czy_jest_blad_lokalny or self.logic_side.czy_jest_blad_lokalny:
                    msg = self.logic_front.komunikat if self.logic_front.czy_jest_blad_lokalny else self.logic_side.komunikat
                    self.feedback_label.configure(text=f"BŁĄD!\n{msg}", fg_color="#c1121f")
                else:
                    msg = self.logic_front.komunikat if self.logic_front.komunikat else "OK"
                    self.feedback_label.configure(text=msg, fg_color="#386641" if msg in ["IDEALNIE", "POSTAWA OK", "DOBRZE"] else "#2b2b2b")
        except Exception as e:
            print(f"[DEBUG] Błąd w update_frame: {e}")
            return

        if self.czy_trening_trwa and self.cam_front and self.cam_side:
            self.update_job = self.after(10, self.update_frame)

    def stop_training(self):
        print("[DEBUG] Rozpoczęcie stop_training...")

        self.czy_trening_trwa = False
        if self.update_job is not None:
            try:
                self.after_cancel(self.update_job)
                self.update_job = None
                print("[DEBUG] Anulowano update_job")
            except Exception as e:
                print(f"[DEBUG] Błąd anulowania update_job: {e}")

        for _ in range(3):
            try:
                self.update_idletasks()
                self.update()
            except:
                pass
            time.sleep(0.05)

        raporty = []
        print("[DEBUG] Generowanie raportów...")
        try:
            if self.logic_front:
                nazwa_raportu = self.logic_front.generuj_raport()
                if nazwa_raportu:
                    raporty.append(nazwa_raportu)
        except Exception as e:
            print(f"[DEBUG] Błąd przy generowaniu raportu FRONT: {e}")
        try:
            if self.logic_side:
                nazwa_raportu = self.logic_side.generuj_raport()
                if nazwa_raportu:
                    raporty.append(nazwa_raportu)
        except Exception as e:
            print(f"[DEBUG] Błąd przy generowaniu raportu SIDE: {e}")

        print("[DEBUG] Zatrzymywanie kamer...")
        try:
            if self.cam_front:
                self.cam_front.stop()
                self.cam_front = None
        except Exception as e:
            print(f"[DEBUG] Błąd przy zatrzymywaniu kamery FRONT: {e}")
        try:
            if self.cam_side:
                self.cam_side.stop()
                self.cam_side = None
        except Exception as e:
            print(f"[DEBUG] Błąd przy zatrzymywaniu kamery SIDE: {e}")

        if raporty:
            print(f"\n{'=' * 60}")
            print(f"SESJA TRENINGOWA ZAKOŃCZONA")
            print(f"{'=' * 60}")
            print(f"Wygenerowano {len(raporty)} raport(ów):")
            for raport in raporty:
                print(f"  • {raport}")
            print(f"{'=' * 60}\n")

        print("[DEBUG] Usuwanie widgetów...")
        for widget in self.winfo_children():
            try:
                widget.destroy()
            except Exception as e:
                print(f"[DEBUG] Błąd przy niszczeniu widgetu: {e}")

        self.logic_front = None
        self.logic_side = None
        time.sleep(0.1)
        try:
            self.update_idletasks()
            self.update()
        except:
            pass

        print("[DEBUG] Powrót do ekranu startowego...")
        self.geometry("600x450")
        self._setup_launcher()
        print("[DEBUG] stop_training zakończone\n")

    def on_closing(self):
        print("[DEBUG] Zamykanie aplikacji...")

        self.czy_trening_trwa = False
        if self.update_job is not None:
            try:
                self.after_cancel(self.update_job)
            except:
                pass

        try:
            if self.cam_front:
                self.cam_front.stop()
        except:
            pass
        try:
            if self.cam_side:
                self.cam_side.stop()
        except:
            pass

        try:
            self.destroy()
        except:
            pass

        print("[DEBUG] Aplikacja zamknięta\n")

    def toggle_skeleton(self):
        self.pokazuj_szkielet = self.skeleton_switch.get() == 1
        self.logic_front.czy_pokazac_szkielet = self.pokazuj_szkielet
        self.logic_side.czy_pokazac_szkielet = self.pokazuj_szkielet

    def change_exercise(self, choice):
        self.aktualne_cwiczenie = choice
        self.exercise_title_label.configure(text=f"SESJA: {choice}")
        self.logic_front = AnalizatorTreningu("FRONT", choice)
        self.logic_side = AnalizatorTreningu(f"SIDE_{self.aktualna_strona}", choice)
        self.logic_front.czy_pokazac_szkielet = self.pokazuj_szkielet
        self.logic_side.czy_pokazac_szkielet = self.pokazuj_szkielet

    def change_side(self, choice):
        self.aktualna_strona = choice
        self.logic_side = AnalizatorTreningu(f"SIDE_{choice}", self.aktualne_cwiczenie)
        self.logic_side.czy_pokazac_szkielet = self.pokazuj_szkielet

    def change_threshold(self, val):
        global PROG_JAKOSCI
        self.prog_jakosci = round(float(val), 2)
        PROG_JAKOSCI = self.prog_jakosci
        self.threshold_label.configure(text=f"Wartość: {self.prog_jakosci}")


# ==============================================================================
# 5. GŁÓWNA PĘTLA PROGRAMU (SYNCHRONIZACJA)
# ==============================================================================
if __name__ == "__main__":
    app = CyberTrenerApp()
    app.mainloop()
