import gradio as gr
import pandas as pd
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

class CrimePredictionApp:
    def __init__(self):
        self.load_data()
        self.load_classifier()

    category_descriptions = {
        "assault": {
            "en": "physical attack or threat of violence against a person including punching, kicking, stabbing, shooting, battery, domestic violence, and threats of harm",
            "de": "Körperverletzung: körperlicher Angriff oder Drohung von Gewalt gegen eine Person einschließlich Schlagen, Treten, Stechen, Schießen, Körperverletzung, häusliche Gewalt und Schadensdrohungen"
        },
        "burglary": {
            "en": "unlawful entry into a building or structure with intent to commit theft, typically when no one is present, including breaking into homes, apartments, stores, warehouses",
            "de": "Einbruch: rechtswidriger Eintritt in ein Gebäude oder eine Struktur mit der Absicht, Diebstahl zu begehen, typischerweise wenn niemand anwesend ist, einschließlich Einbruch in Häuser, Wohnungen, Geschäfte, Lagerhäuser"
        },
        "larceny/theft": {
            "en": "unlawful taking of property without force or breaking entry, including shoplifting, pickpocketing, purse snatching, bicycle theft, and theft from vehicles",
            "de": "Diebstahl: rechtswidriger Wegnahme von Eigentum ohne Gewalt oder Einbruch, einschließlich Ladendiebstahl, Taschendiebstahl, Handtaschendiebstahl, Fahrraddiebstahl und Diebstahl von Fahrzeugen"
        },
        "robbery": {
            "en": "taking property directly from a person using force, intimidation, or threat of violence, including armed robbery, mugging, bank robbery, and carjacking",
            "de": "Raub: direkter Wegnahme von Eigentum von einer Person unter Anwendung von Gewalt, Einschüchterung oder Drohung von Gewalt, einschließlich bewaffnetem Raubüberfall, Straßenraub, Bankraub und Carjacking"
        },
        "fraud": {
            "en": "intentional deception for financial gain, including identity theft, credit card fraud, forgery, counterfeiting, false pretenses, check fraud, and fraudulent financial transactions",
            "de": "Betrug: vorsätzliche Täuschung zum finanziellen Vorteil, einschließlich Identitätsdiebstahl, Kreditkartenbetrug, Fälschung, Falschung, falsche Vortäuschungen, Scheckbetrug und betrügerische Finanztransaktionen"
        },
        "property damage": {
            "en": "deliberate destruction or defacement of property, including vandalism, graffiti, malicious mischief, smashing windows, slashing tires, and keying vehicles",
            "de": "Sachbeschädigung: vorsätzliche Zerstörung oder Beschädigung von Eigentum, einschließlich Vandalismus, Graffiti, bösartige Sachbeschädigung, Zerschlagen von Fenstern, Zerschneiden von Reifen und Aufschließen von Fahrzeugen"
        },
        "driving under the influence": {
            "en": "operating a vehicle while impaired by alcohol or drugs, including DUI, DWI, reckless driving while intoxicated, and related traffic offenses",
            "de": "Trunkenheit am Steuer: Fahren eines Fahrzeugs unter Einfluss von Alkohol oder Drogen, einschließlich DUI, DWI, rücksichtsloses Fahren unter Einfluss und damit verbundene Verkehrsverstöße"
        },
        "disorderly conduct": {
            "en": "public disturbance behavior including public intoxication, fighting in public, loud disturbances, trespassing, loitering, and public nuisance",
            "de": "Ruhestörung: störendes Verhalten in der Öffentlichkeit, einschließlich öffentlicher Trunkenheit, öffentliche Kämpfe, laute Störungen, unerlaubtes Betreten, Herumlung und öffentliche Belästigung"
        },
        "missing person": {
            "en": "reports of individuals whose whereabouts are unknown, including missing adults, missing juveniles, runaways, and found persons",
            "de": "Vermisste Person: Meldungen über Personen, deren Aufenthaltsort unbekannt ist, einschließlich vermisste Erwachsene, vermisste Minderjährige, Ausreißende und gefundene Personen"
        },
        "drug/narcotic": {
            "en": "offenses involving illegal drugs including possession, sale, trafficking, manufacturing of controlled substances such as cocaine, heroin, marijuana, and methamphetamine",
            "de": "Drogen/Verstöße: Verstöße, die illegale Drogen betreffen, einschließlich Besitz, Verkauf, Handel, Herstellung kontrollierter Substanzen wie Kokain, Heroin, Marihuana und Methamphetamin"
        },
        "vehicle theft": {
            "en": "unlawful taking of a motor vehicle, including stolen automobiles, trucks, motorcycles, and recovered stolen vehicles",
            "de": "Fahrzeugdiebstahl: rechtswidriger Wegnahme eines Kraftfahrzeugs, einschließlich gestohlener Automobile, Lastwagen, Motorräder und wiederhergestellte gestohlene Fahrzeuge"
        },
        "weapon laws": {
            "en": "illegal possession, carrying, or use of weapons including concealed firearms, switchblades, assault weapons, and violations of firearm regulations",
            "de": "Waffengesetze: illegaler Besitz, Tragen oder Gebrauch von Waffen, einschließlich versteckter Schusswaffen, Springmesser, Angriffswaffen und Verstöße gegen Schusswaffenverordnungen"
        },
        "embezzlement": {
            "en": "misappropriation of funds or property entrusted to someone, including employee theft, theft by fiduciary, theft by public official, and corporate fund misuse",
            "de": "Untreue: rechtswidrige Aneignung von Geldern oder Eigentum, das jemandem anvertraut wurde, einschließlich Diebstahl durch Arbeitnehmer, Diebstahl durch Treuhänder, Diebstahl durch Beamte und missbräuchliche Verwendung von Unternehmensmitteln"
        },
        "kidnapping": {
            "en": "unlawful seizure and detention of a person against their will, including abduction of adults or children, hostage situations, and kidnapping during other crimes",
            "de": "Entführung: rechtswidrige Festnahme und Inhaftierung einer Person gegen ihren Willen, einschließlich Entführung von Erwachsenen oder Kindern, Geiselnahmen und Entführung während anderer Verbrechen"
        },
        "arson": {
            "en": "intentional and malicious setting of fire to property, including burning buildings, vehicles, dwellings, and attempted arson",
            "de": "Brandstiftung: vorsätzliche und bösartige Inbrandsetzung von Eigentum, einschließlich Abbrennen von Gebäuden, Fahrzeugen, Wohnungen und versuchte Brandstiftung"
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
            self.classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
            
            # Store categories as dict with separate language keys
            self.candidate_labels = []
            for category_key, descriptions in self.category_descriptions.items():
                self.candidate_labels.append(f"{category_key.title()}: {descriptions['en']}")
                self.candidate_labels.append(f"{category_key.title()} / {descriptions['de']}")
            print("Zero-shot classifier loaded successfully!")
            
        except Exception as e:
            print(f"Error loading classifier: {e}")
            self.classifier = None
    
    def predict_crime(self, text):
        """Predict crime category using zero-shot classification enhanced with keyword matching"""
        if not text.strip():
            return "Please enter a valid description", "No prediction available", 0.0
        
        if self.classifier is None:
            return self._keyword_prediction(text)
            
        try:
            # Use zero-shot classification with candidate labels
            output = self.classifier(text, self.candidate_labels, multi_label=True)
            hits = [
                f"{label} ({score:.1%})\n" 
                for label, score in zip(output['labels'], output['scores']) 
                if score > 0.85
            ]
            if hits:
                best_idx = output['scores'].index(max(output['scores']))
                prediction = output['labels'][best_idx]
                confidence = max(output['scores'])
                prediction_details = "\n".join(hits)
                return prediction, prediction_details, confidence
            else:
                # No high-confidence zero-shot matches, use keyword prediction
                return self._keyword_prediction(text)
                
        except Exception as e:
            print(f"Error in zero-shot classification: {e}")
            return self._keyword_prediction(text)
    
    def _keyword_prediction(self, text):
        """Predict crime category using keyword matching"""
        text_lower = text.lower()
        if any(word in text_lower for word in ['theft', 'stolen', 'larceny']):
            prediction = 'larceny/theft'
        elif any(word in text_lower for word in ['assault', 'battery', 'violence', 'attack', 'fight']):
            prediction = 'assault'
        elif any(word in text_lower for word in ['drug', 'narcotic']):
            prediction = 'drug/narcotic'
        elif any(word in text_lower for word in ['vehicle', 'auto', 'car']):
            prediction = 'vehicle theft'
        elif any(word in text_lower for word in ['vandalism', 'damage', 'graffiti']):
            prediction = 'vandalism'
        elif any(word in text_lower for word in ['burglary', 'break', 'window', 'climbing']):
            prediction = 'burglary'
        elif any(word in text_lower for word in ['robbery', 'mask', 'grab']):
            prediction = 'robbery'
        elif any(word in text_lower for word in ['weapon', 'knife', 'gun']):
            prediction = 'weapon laws'
        elif any(word in text_lower for word in ['fraud', 'fake', 'counterfeit']):
            prediction = 'fraud'
        elif any(word in text_lower for word in ['forgery', 'fake']):
            prediction = 'forgery/counterfeiting'
        elif any(word in text_lower for word in ['bribery']):
            prediction = 'bribery'
        elif any(word in text_lower for word in ['embezzlement']):
            prediction = 'embezzlement'
        elif any(word in text_lower for word in ['extortion']):
            prediction = 'extortion'
        elif any(word in text_lower for word in ['kidnapping']):
            prediction = 'kidnapping'
        elif any(word in text_lower for word in ['missing person']):
            prediction = 'missing person'
        elif any(word in text_lower for word in ['family offenses']):
            prediction = 'family offenses'
        elif any(word in text_lower for word in ['loitering']):
            prediction = 'loitering'
        elif any(word in text_lower for word in ['disorderly conduct']):
            prediction = 'disorderly conduct'
        elif any(word in text_lower for word in ['drunkenness', 'drunk']):
            prediction = 'drunkenness'
        elif any(word in text_lower for word in ['driving under the influence', 'dui', 'drunk driving']):
            prediction = 'driving under the influence'
        elif any(word in text_lower for word in ['liquor laws']):
            prediction = 'liquor laws'
        elif any(word in text_lower for word in ['gambling']):
            prediction = 'gambling'
        elif any(word in text_lower for word in ['arson', 'fire']):
            prediction = 'arson'
        elif any(word in text_lower for word in ['sex offenses', 'sexual']):
            prediction = 'sex offenses forcible'
        elif any(word in text_lower for word in ['prostitution']):
            prediction = 'prostitution'
        elif any(word in text_lower for word in ['pornography', 'obscene']):
            prediction = 'pornography/obscene mat'
        elif any(word in text_lower for word in ['trespass']):
            prediction = 'trespass'
        elif any(word in text_lower for word in ['stolen property']):
            prediction = 'stolen property'
        elif any(word in text_lower for word in ['recovered vehicle']):
            prediction = 'recovered vehicle'
        elif any(word in text_lower for word in ['warrants']):
            prediction = 'warrants'
        elif any(word in text_lower for word in ['suicide']):
            prediction = 'suicide'
        elif any(word in text_lower for word in ['suspicious', 'susp']):
            prediction = 'suspicious occ'
        elif any(word in text_lower for word in ['runaway']):
            prediction = 'runaway'
        elif any(word in text_lower for word in ['non-criminal']):
            prediction = 'non-criminal'
        elif any(word in text_lower for word in ['secondary codes']):
            prediction = 'secondary codes'
        else:
            prediction = 'other offenses'  # Default fallback
        
        valid_categories = self.unique_categories
        # If prediction is not in valid categories, set it to 'other offenses'
        if prediction not in valid_categories:
            prediction = 'other offenses'
        confidence = 0.7
        prediction_details = f"{prediction}: {confidence:.0%}\nKeyword-based prediction using valid categories"
        return prediction, prediction_details, confidence
    
    def get_crime_statistics(self):
        """Get crime statistics: dataset size, available categories, zero-shot model status"""
        try:
            total_descriptions = len(self.crime_data)
            stats_text = f"• Total Crime Descriptions: {total_descriptions:,}\n"
            stats_text += f"• Available Categories: {len(self.unique_categories)}\n"
            stats_text += f"• Zero-Shot Model: {'✅ Active' if self.classifier else '❌ Inactive'}\n\n"
            stats_text += "Available Categories:\n"
            for i, category in enumerate(self.unique_categories[:10], 1):
                stats_text += f"{i}. {category}\n"
            if len(self.unique_categories) > 10:
                stats_text += f"... and {len(self.unique_categories) - 10} more\n"
            return stats_text
        except Exception as e:
            return f"Error generating statistics: {e}"

# Initialize the app
app = CrimePredictionApp()

def predict_interface(input_text):
    """Predict crime category based on a given crime description"""
    prediction, details, confidence = app.predict_crime(input_text)
    if confidence == 0.0:
        return prediction, details, "⚠️ Please enter a valid crime description"
    confidence_emoji = "🔴" if confidence < 0.3 else "🟡" if confidence < 0.6 else "🟢"
    confidence_text = f"{confidence_emoji} Confidence: {confidence:.1%}"
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
            gr.Markdown("### 📊 Quick Statistics")
            stats_display = gr.Textbox(
                value=app.get_crime_statistics(),
                label="Dataset Overview",
                lines=10,
                interactive=False
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
    
    gr.Examples(
        examples=[
            # English Examples
            ["A person stole a wallet from someone's pocket on the bus"],
            ["Someone broke the window of a store and took merchandise"],
            ["Two people were fighting in the street and one was injured"],
            ["Police found illegal drugs during a traffic stop"],
            ["Someone spray-painted graffiti on the school building"],
            ["A car was stolen from the parking lot overnight"],
            ["Person threatened another with a knife during an argument"],
            # German Examples
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
    
    input_text.submit(
        fn=predict_interface,
        inputs=input_text,
        outputs=[prediction_output, details_output, confidence_output]
    )

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)