"""Step 3: Tkinter GUI. Run this file:  python gui.py"""
import io
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

from cleandata import load_movies
from compare import hand_test_score
from recommender import MovieRecommender, NoMatchError
from scraper import get_poster_bytes

TFIDF, SPACY, HYBRID = "TF-IDF + KNN", "spaCy vectors", "Hybrid (TF-IDF + spaCy)"
MODEL_NAMES = [TFIDF, SPACY, HYBRID]
DESCRIPTIONS = {
    TFIDF: "TF-IDF + K-Nearest Neighbors (cosine, k = 5)\n   matches exact words (text = overview + keywords + tagline)",
    SPACY: "spaCy word vectors (en_core_web_md) + cosine similarity\n   matches meaning, e.g. 'extinct animals' ~ 'dinosaurs'",
    HYBRID: "50% TF-IDF score + 50% spaCy score",
}


class CineMatchApp:
    def __init__(self, root):
        self.root = root
        root.title("CineMatch")
        root.geometry("950x580")

        self.tfidf = None          # always built: also used for the k-means themes
        self.models = {}           # name -> model, built the first time it is selected
        self.recommender = None    # the model currently in use
        self.model_name = TFIDF
        self.results = None
        self.photo = None          # keep a reference, otherwise Tkinter drops the image
        self.accuracy = None
        self.hand_hits = (0, 0)

        # ---- top bar ----
        top = ttk.Frame(root, padding=(10, 10, 10, 0))
        top.pack(fill="x")
        ttk.Label(top, text="Describe the plot:").pack(side="left")
        self.entry = ttk.Entry(top, width=50)
        self.entry.pack(side="left", padx=8)
        self.entry.bind("<Return>", lambda event: self.search())
        ttk.Button(top, text="Search", command=self.search).pack(side="left")
        ttk.Button(top, text="Model info", command=self.show_model_info).pack(side="left", padx=8)

        bar2 = ttk.Frame(root, padding=(10, 6, 10, 0))
        bar2.pack(fill="x")
        ttk.Label(bar2, text="Model:").pack(side="left")
        self.model_box = ttk.Combobox(bar2, values=MODEL_NAMES, state="readonly", width=26)
        self.model_box.set(TFIDF)
        self.model_box.pack(side="left", padx=8)
        self.model_box.bind("<<ComboboxSelected>>", lambda event: self.activate(self.model_box.get()))
        self.theme_label = ttk.Label(bar2, text="", wraplength=420)
        self.theme_label.pack(side="left", padx=15)

        # ---- results list (left) ----
        body = ttk.Frame(root, padding=10)
        body.pack(fill="both", expand=True)
        self.listbox = tk.Listbox(body, width=34, height=15, exportselection=False)
        self.listbox.pack(side="left", fill="y")
        self.listbox.bind("<<ListboxSelect>>", self.show_movie)

        # ---- details (right) ----
        self.poster_label = ttk.Label(body, text="Poster appears here", width=28)
        self.poster_label.pack(side="left", anchor="n", padx=15)
        info = ttk.Frame(body)
        info.pack(side="left", fill="both", expand=True)
        self.title_label = ttk.Label(info, font=("Arial", 15, "bold"), wraplength=380)
        self.title_label.pack(anchor="w")
        self.meta_label = ttk.Label(info, text="")
        self.meta_label.pack(anchor="w", pady=4)
        self.overview = tk.Text(info, wrap="word", height=14, width=48, state="disabled")
        self.overview.pack(fill="both", expand=True)

        self.status = ttk.Label(root, text="Loading data and models (the first spaCy run takes a few minutes)...", relief="sunken", anchor="w")
        self.status.pack(fill="x", side="bottom")

        root.after(100, self.load_models)

    # ---------- models ----------
    def load_models(self):
        try:
            df = load_movies()
        except FileNotFoundError as error:
            messagebox.showerror("Data missing", str(error))
            self.root.destroy()
            return
        self.tfidf = MovieRecommender(df, mode="lemma", text_column="full_text", sublinear_tf=True)
        self.tfidf.build_clusters(15)
        self.models[TFIDF] = self.tfidf
        if not self.activate(HYBRID, quiet=True):      # best model; needs spaCy
            self.activate(TFIDF)                       # fallback

    def get_model(self, name):
        """Return the model, building it the first time (spaCy vectors take a while)."""
        if name in self.models:
            return self.models[name]
        from spacy_recommender import SpacyRecommender, HybridRecommender   # ImportError if spaCy is missing
        self.status.config(text="Building spaCy vectors (first time takes a few minutes)...")
        self.root.update_idletasks()
        if SPACY not in self.models:
            self.models[SPACY] = SpacyRecommender(self.tfidf.df)
        if name == HYBRID:
            self.models[HYBRID] = HybridRecommender(self.tfidf, self.models[SPACY], weight=0.5)
        return self.models[name]

    def activate(self, name, quiet=False):
        """Switch to a model. Returns True on success, False if it could not be built."""
        try:
            model = self.get_model(name)
        except (ImportError, RuntimeError) as error:
            if not quiet:
                messagebox.showerror("Model not available", f"{error}\n\nInstall:  pip install spacy")
            self.model_box.set(self.model_name)
            self.status.config(text=f"Using {self.model_name}.")
            return False
        self.recommender = model
        self.model_name = name
        self.model_box.set(name)
        self.status.config(text="Testing the model...")
        self.root.update_idletasks()
        self.accuracy = model.evaluate()
        self.hand_hits = hand_test_score(model)
        self.status.config(
            text=f"Ready. {len(self.tfidf.df)} movies  |  Model: {name}  |  Top-5 accuracy: {self.accuracy:.0%}"
        )
        return True

    def show_model_info(self):
        if self.recommender is None:
            messagebox.showinfo("Model info", "Still loading, please wait...")
            return
        hits, total = self.hand_hits
        hand = f"{hits} of {total} in the top 5" if total else "no test movies found in the data"
        messagebox.showinfo(
            "Model info",
            f"Recommendation model: {self.model_name}\n"
            f"   {DESCRIPTIONS[self.model_name]}\n"
            "Theme model: k-means (15 clusters)\n"
            f"Movies: {len(self.tfidf.df)}\n\n"
            f"Top-5 accuracy: {self.accuracy:.1%}\n"
            "   For 500 random movies, a query made of half the words of\n"
            "   its description. Correct if that movie is in the top 5.\n\n"
            f"Hand-written test queries: {hand}",
        )

    # ---------- searching ----------
    def search(self):
        if self.recommender is None:
            self.status.config(text="Still loading, please wait...")
            return
        query = self.entry.get()
        try:
            self.results = self.recommender.recommend(query, top_n=5)
        except (ValueError, NoMatchError) as error:
            messagebox.showinfo("CineMatch", str(error))
            return

        self.listbox.delete(0, "end")
        for _, row in self.results.iterrows():
            self.listbox.insert("end", f"{row['title']}  ({row['similarity']:.2f})")
        theme = self.tfidf.theme_of(query)
        self.theme_label.config(text=f"Theme: {theme}" if theme else "")
        self.listbox.selection_set(0)
        self.show_movie()

    def show_movie(self, event=None):
        selected = self.listbox.curselection()
        if not selected:
            return
        row = self.results.iloc[selected[0]]

        self.title_label.config(text=row["title"])
        self.meta_label.config(text=f"Match score: {row['similarity']:.2f}")
        self.overview.config(state="normal")
        self.overview.delete("1.0", "end")
        self.overview.insert("end", row["overview"])
        self.overview.config(state="disabled")

        self.poster_label.config(image="", text="Loading poster...")
        self.root.update_idletasks()
        data = get_poster_bytes(row["title"])
        if data is None:
            self.photo = None
            self.poster_label.config(image="", text="No poster found")
        else:
            image = Image.open(io.BytesIO(data))
            image.thumbnail((200, 300))
            self.photo = ImageTk.PhotoImage(image)
            self.poster_label.config(image=self.photo, text="")


if __name__ == "__main__":
    window = tk.Tk()
    CineMatchApp(window)
    window.mainloop()