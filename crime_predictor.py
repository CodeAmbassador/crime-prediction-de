import gradio as gr
import pandas as pd
from langdetect import detect
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

class CrimePredictionApp:
    """Crime prediction APP using zero-shot classification"""

    def __init__(self):
        self.load_data()
        self.load_classifier()

    category_descriptions = {
        "assault": {
            "en": "physical attack or threat of violence against a person including punching, kicking, stabbing, shooting, battery, domestic violence, and threats of harm",
            "de": "Körperverletzungs- und Bedrohungsdelikte: vorsätzliche körperliche Misshandlung oder Gesundheitsschädigung einer Person (z.B. Schlagen, Treten, Stechen, Schüsse) sowie Drohungen mit Gewalt oder mit einem Verbrechen (§§ 223 ff., 240, 241 StGB), einschließlich häuslicher Gewalt."
        },
        "burglary": {
            "en": "unlawful entry into a building or structure with intent to commit theft, typically when no one is present, including breaking into homes, apartments, stores, warehouses",
            "de": "Einbruchdiebstahl: unbefugtes Eindringen in ein Gebäude oder einen umschlossenen Raum zur Begehung eines Diebstahls (§§ 242, 243, 244 StGB), typischerweise wenn niemand anwesend ist, z.B. Einbrüche in Häuser, Wohnungen, Geschäfte oder Lagerhallen."
        },
        "larceny/theft": {
            "en": "unlawful taking of property without force or breaking entry, including shoplifting, pickpocketing, purse snatching, bicycle theft, and theft from vehicles",
            "de": "Diebstahl ohne Einbruch: rechtswidrige Wegnahme einer fremden beweglichen Sache ohne Gewaltanwendung gegen Personen, ohne Einbruch und ohne Kfz-Diebstahl (§ 242 StGB), z.B. Ladendiebstahl, Taschendiebstahl, Handtaschendiebstahl, Fahrraddiebstahl oder Diebstahl aus Fahrzeugen."
        },
        "robbery": {
            "en": "taking property directly from a person using force, intimidation, or threat of violence, including armed robbery, mugging, bank robbery, and carjacking",
            "de": "Raub: Wegnahme einer fremden beweglichen Sache unter Anwendung von Gewalt gegen eine Person oder Drohung mit gegenwärtiger Gefahr für Leib oder Leben (§§ 249 f. StGB), z.B. bewaffneter Raubüberfall, Straßenraub, Bankraub oder Carjacking."
        },
        "fraud": {
            "en": "intentional deception for financial gain, including identity theft, credit card fraud, forgery, counterfeiting, false pretenses, check fraud, and fraudulent financial transactions",
            "de": "Betrugsdelikte: vorsätzliche Täuschung zur Erlangung eines rechtswidrigen Vermögensvorteils (§§ 263 ff. StGB), z.B. Identitätsbetrug, Kreditkartenbetrug, Betrug durch falsche Vorspiegelung von Tatsachen, Computerbetrug, Scheckbetrug und sonstige betrügerische Finanztransaktionen (ggf. im Zusammenspiel mit Urkundenfälschung, § 267 StGB)."
        },
        "property damage": {
            "en": "deliberate destruction or defacement of property, including vandalism, graffiti, malicious mischief, smashing windows, slashing tires, and keying vehicles",
            "de": "Sachbeschädigung: vorsätzliche oder gemeingefährliche Beschädigung oder Zerstörung fremder Sachen (§§ 303, 304 StGB), z.B. Vandalismus, Graffiti, mutwilliges Zerschlagen von Fenstern, Zerkratzen von Fahrzeugen („Keying“) oder Zerstechen von Reifen."
        },
        "driving under the influence": {
            "en": "operating a vehicle while impaired by alcohol or drugs, including DUI, DWI, reckless driving while intoxicated, and related traffic offenses",
            "de": "Trunkenheit im Verkehr: Führen eines Fahrzeugs im Straßenverkehr unter erheblichem Einfluss von Alkohol oder anderen berauschenden Mitteln (§§ 316, 315c StGB), einschließlich Fahrens unter Alkohol- oder Drogeneinfluss sowie damit verbundener gefährlicher oder rücksichtsloser Fahrweisen."
        },
        "disorderly conduct": {
            "en": "public disturbance behavior including public intoxication, fighting in public, loud disturbances, trespassing, loitering, and public nuisance",
            "de": "Störung der öffentlichen Ordnung: belästigendes oder aggressives Verhalten im öffentlichen Raum, das die Allgemeinheit oder die öffentliche Sicherheit beeinträchtigt (insbesondere Ordnungswidrigkeiten nach § 118 OWiG, je nach Schwere auch Straftaten nach dem StGB), z.B. öffentliche Trunkenheit, Raufereien, lautstarke Ruhestörungen, unbefugtes Betreten von Grundstücken/Gebäuden (Hausfriedensbruch), aggressives Herumlungern und sonstige öffentliche Belästigungen."
        },
        "missing person": {
            "en": "reports of individuals whose whereabouts are unknown, including missing adults, missing juveniles, runaways, and found persons",
            "de": "Vermisstenfälle: polizeiliche Meldungen über Personen, deren Aufenthaltsort unbekannt ist, einschließlich vermisster Erwachsener, vermisster Minderjähriger, mutmaßlicher Ausreißer sowie wiederaufgefundener Personen (kein eigenständiger Straftatbestand)."
        },
        "drug/narcotic": {
            "en": "offenses involving illegal drugs including possession, sale, trafficking, manufacturing of controlled substances such as cocaine, heroin, marijuana, and methamphetamine",
            "de": "Betäubungsmitteldelikte: Straftaten im Zusammenhang mit unerlaubten Betäubungsmitteln nach dem Betäubungsmittelgesetz (BtMG), insbesondere unerlaubter Besitz, Erwerb, Handel, Einfuhr, Ausfuhr, Abgabe, Veräußerung oder Herstellung von Stoffen wie Kokain, Heroin, Cannabis oder Methamphetamin."
        },
        "vehicle theft": {
            "en": "unlawful taking of a motor vehicle, including stolen automobiles, trucks, motorcycles, and recovered stolen vehicles",
            "de": "Fahrzeugdiebstahl / unbefugter Gebrauch: rechtswidrige Wegnahme oder unbefugte Ingebrauchnahme eines Kraftfahrzeugs (§§ 242, 248b StGB), z.B. Diebstahl von Pkw, Lkw oder Motorrädern sowie das Auffinden und Sicherstellen gestohlener Fahrzeuge."
        },
        "weapon laws": {
            "en": "illegal possession, carrying, or use of weapons including concealed firearms, switchblades, assault weapons, and violations of firearm regulations",
            "de": "Waffenrechtsverstöße: Verstöße gegen das Waffengesetz (WaffG) oder vergleichbare Vorschriften, insbesondere unerlaubter Erwerb, Besitz, Führen oder Gebrauch von Schusswaffen, verbotenen Messern (z.B. Springmessern) oder anderen verbotenen Waffen sowie Zuwiderhandlungen gegen Aufbewahrungs- und Erlaubnispflichten."
        },
        "embezzlement": {
            "en": "misappropriation of funds or property entrusted to someone, including employee theft, theft by fiduciary, theft by public official, and corporate fund misuse",
            "de": "Untreue / Veruntreuung: missbräuchliche Verwendung oder rechtswidrige Zueignung von Vermögenswerten, die dem Täter anvertraut wurden (§§ 266, 246 StGB), z.B. Veruntreuung von Firmengeldern durch Arbeitnehmer, Veruntreuung durch Treuhänder oder Bevollmächtigte, Unterschlagung von Kundengeldern oder Vermögen durch Amtsträger."
        },
        "kidnapping": {
            "en": "unlawful seizure and detention of a person against their will, including abduction of adults or children, hostage situations, and kidnapping during other crimes",
            "de": "Freiheitsberaubungs- und Entführungsdelikte: unrechtmäßiges Festhalten oder Entführen einer Person gegen ihren Willen, z.B. Freiheitsberaubung (§ 239 StGB), Entziehung Minderjähriger (§ 235 StGB), erpresserischer Menschenraub (§ 239a StGB) oder Geiselnahme (§ 239b StGB), einschließlich Entführungen von Erwachsenen oder Kindern und Geisellagen."
        },
        "arson": {
            "en": "intentional and malicious setting of fire to property, including burning buildings, vehicles, dwellings, and attempted arson",
            "de": "Brandstiftungsdelikte: vorsätzliches oder in bestimmten Fällen fahrlässiges Inbrandsetzen oder Zerstören von Gebäuden, Fahrzeugen oder anderen Objekten durch Feuer (§§ 306 ff. StGB), einschließlich Brandlegung in Wohn- und Geschäftsgebäuden, an Kraftfahrzeugen sowie versuchter Brandstiftung."
        }
    }
    
    def load_data(self):
        """Load and prepare the crime dataset"""
        try:
            descriptions_df = pd.read_csv("./dataset/descript_en.csv")
            self.crime_data = descriptions_df[['Descript']].copy()
            categories_df = pd.read_csv("./dataset/category_en.csv")
            self.unique_categories = sorted(categories_df['Category'].unique())
            print(f"Loaded {len(self.crime_data)} descriptions from descript_en.csv")
            print(f"Loaded {len(self.unique_categories)} valid categories from category_en.csv")
            
        except FileNotFoundError:
            print("Error: Dataset files (descript_en.csv or category_en.csv) not found. Please ensure the dataset is in the correct location.")
            exit()
    
    def load_classifier(self):
        """Load zero-shot classification model"""
        try:
            print("Loading zero-shot classification model...")
            # Load model with CPU optimizations
            self.classifier = pipeline(
                "zero-shot-classification", 
                model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
                device=-1,  # Force CPU usage
                dtype="auto",
                model_kwargs={
                    "low_cpu_mem_usage": True
                }
            )
            self.candidate_labels_en = [desc["en"] for desc in CrimePredictionApp.category_descriptions.values()]
            self.candidate_labels_de = [desc["de"] for desc in CrimePredictionApp.category_descriptions.values()]
            self.sf_categories = self.unique_categories
            print("Zero-shot classifier loaded successfully!")
            
        except Exception as e:
            print(f"Error loading classifier: {e}")
            self.classifier = None
    
    
    def predict_crime(self, text):
        """Predicts crime category using zero-shot classification with language detection"""
        if not text.strip() or self.classifier is None:
            return "No prediction", "Please enter a valid description", 0.0, ""
        detected_lang = detect(text)
        candidate_labels = self.candidate_labels_de if detected_lang == 'de' else self.candidate_labels_en
        print(f"Detected language: {detected_lang.upper()}")

        try:
            output = self.classifier(text, candidate_labels, multi_label=True)
            # Get top 3 predictions with confidence scores
            top_predictions = []
            for desc_text, score in zip(output['labels'], output['scores']):
                if score > 0.1:
                    category_key = None
                    for key, desc_dict in self.category_descriptions.items():
                        if desc_dict[detected_lang] == desc_text:
                            category_key = key
                            break
                    if category_key:
                        desc = self.category_descriptions[category_key][detected_lang]
                        top_predictions.append((category_key, score, desc))
            
            top_predictions.sort(key=lambda x: x[1], reverse=True)
            top_3 = top_predictions[:3]
            if not top_3:
                return "No Match", "Could not classify description", 0.0, ""
            details = ""
            for i, (label, score, desc) in enumerate(top_3, 1):
                confidence_emoji = "🔴" if score < 0.3 else "🟡" if score < 0.6 else "🟢"
                details += f"{i}. {label.upper()} {confidence_emoji} {score:.1%}\n"
                details += f"   {desc}\n\n"
            best_match, confidence, _ = top_3[0]
            confidence_emoji = "🔴" if confidence < 0.3 else "🟡" if confidence < 0.6 else "🟢"
            return best_match, details, confidence, confidence_emoji

        except Exception as e:
            print(f"Prediction Error: {e}")
            return "No Match", "Could not classify the description", 0.0, ""
    
    def get_crime_statistics(self):
        """Get crime statistics: dataset size, zero-shot model status"""
        try:
            total_descriptions = len(self.crime_data)
            stats_text = f"• Zero-Shot Model: {'✅ Active' if self.classifier else '❌ Inactive'}\n\n"
            stats_text += f"• Total Crime Descriptions: {total_descriptions:,}\n\n"
            stats_text += f"• Available Categories ({len(self.unique_categories)}):\n"
            for i, category in enumerate(self.unique_categories, 1):
                stats_text += f"{i}. {category}\n"
            return stats_text

        except Exception as e:
            return f"Error generating statistics: {e}"

# Initialize the app
app = CrimePredictionApp()

def predict_interface(input_text):
    """Predict crime category based on a given crime description"""
    prediction, details, confidence, confidence_emoji = app.predict_crime(input_text)
    if confidence == 0.0:
        return prediction, details, "⚠️ Please enter a valid crime description"
    confidence_text = f"{confidence_emoji} {confidence:.1%}"
    return prediction, details, confidence_text

# Create the Gradio interface
with gr.Blocks(title="Crime Prediction System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🚔 Crime Prediction System
    
    Enter a crime description to get a crime category prediction based on San Francisco crime data from 1934 to 1963 (Origin: https://www.kaggle.com/competitions/sf-crime/overview).
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                label="Crime Description (Multilanguage Support)",
                placeholder="Describe the crime incident (see some Examples below!).",
                lines=3,
                max_lines=20
            )
            predict_btn = gr.Button("🔍 Predict Crime Category", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            with gr.Accordion("📊 Setup", open=False):
                stats_display = gr.Markdown(
                    app.get_crime_statistics()
            )
    
    with gr.Row():
        with gr.Column():
            prediction_output = gr.Textbox(
                label="Predicted Category",
                interactive=False,
                lines=2
            )
        
        with gr.Column():
            confidence_output = gr.Textbox(
                label="Confidence Level",
                interactive=False,
                lines=1
            )
    
    with gr.Row():
        details_output = gr.Textbox(
            label="Top Predictions (with probabilities)",
            interactive=False,
            lines=20
        )
    
    # Input examples
    gr.Examples(
        examples=[
            # English
            ["A person stole a wallet from someone's pocket on the bus"],
            ["Someone broke the window of a store and took merchandise"],
            ["Two people were fighting in the street and one was injured"],
            ["Police found illegal drugs during a traffic stop"],
            ["Someone spray-painted graffiti on the school building"],
            ["A car was stolen from the parking lot overnight"],
            ["Person threatened another with a knife during an argument"],
            # German
            ["Ein Mann klaute ein Fahrrad vor dem Supermarkt"],
            ["Jemand warf einen Stein durch die Autowindschutzscheibe"],
            ["Gruppe Jugendlicher prügelten sich am Bahnhof"],
            ["Polizei entdeckte Marihuana bei einer Personenkontrolle"],
            ["Unbekannter sprühte 'Freiheit' auf die Stadtmauer"],
        ],
        inputs=input_text,
        outputs=[prediction_output, details_output, confidence_output],
        fn=predict_interface,
        examples_per_page=100
    )
    
    # Event handlers
    predict_btn.click(
        fn=predict_interface,
        inputs=input_text,
        outputs=[prediction_output, details_output, confidence_output]
    )
    input_text.submit(
        fn=predict_interface,
        inputs=input_text,
        outputs=[prediction_output, details_output, confidence_output]
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)