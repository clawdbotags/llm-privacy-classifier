# German language test cases for privacy classifier
# Mix of SENSITIVE and PUBLIC in German

GERMAN_CASES = [
    # SENSITIVE: Personal identifiers
    ("Meine Sozialversicherungsnummer ist 12 150385 K 042, können Sie mir bei der Steuererklärung helfen?", "SENSITIVE"),
    ("Ich bin am 14. März 1987 geboren und muss meine Identität bestätigen.", "SENSITIVE"),
    ("Meine Reisepassnummer ist C01X00T47, ist er noch gültig für die Reise nach Japan?", "SENSITIVE"),
    ("Mein Personalausweis hat die Nummer T220001293, wie erneuere ich ihn?", "SENSITIVE"),

    # SENSITIVE: Contact info
    ("Meine E-Mail ist hans.mueller@gmail.com, schicken Sie den Bericht bitte dorthin.", "SENSITIVE"),
    ("Erreichen Sie mich unter 0176 12345678 nach 17 Uhr.", "SENSITIVE"),
    ("Ich wohne in der Hauptstraße 42, 80331 München.", "SENSITIVE"),
    ("Bitte senden Sie das Paket an Schillerstraße 15, 10625 Berlin, z.Hd. Frau Schmidt.", "SENSITIVE"),

    # SENSITIVE: Financial
    ("Mein Gehalt beträgt 65.000 Euro im Jahr und ich möchte verhandeln.", "SENSITIVE"),
    ("Meine IBAN ist DE89 3704 0044 0532 0130 00, bitte überweisen Sie den Betrag.", "SENSITIVE"),
    ("Ich habe 23.000 Euro Schulden bei der Sparkasse.", "SENSITIVE"),
    ("Mein Schufa-Score ist auf 78 gefallen nach der versäumten Zahlung.", "SENSITIVE"),
    ("Wir verdienen zusammen 120.000 Euro brutto, reicht das für ein Haus in München?", "SENSITIVE"),

    # SENSITIVE: Health
    ("Mein Arzt hat bei mir Typ-2-Diabetes diagnostiziert.", "SENSITIVE"),
    ("Ich nehme seit drei Monaten Sertralin 50mg gegen meine Depressionen.", "SENSITIVE"),
    ("Mein Blutdruck war beim letzten Arztbesuch bei Dr. Müller 170/100.", "SENSITIVE"),
    ("Mein Sohn hat ADHS und die Schule verlangt ein Gutachten.", "SENSITIVE"),
    ("Ich war letzte Woche in der psychiatrischen Klinik.", "SENSITIVE"),

    # SENSITIVE: Legal/Private
    ("Ich lasse mich scheiden, mein Mann hat eine Affäre.", "SENSITIVE"),
    ("Mein Chef Thomas hat mich letzte Woche belästigt.", "SENSITIVE"),
    ("Ich wurde wegen Trunkenheit am Steuer angehalten und brauche einen Anwalt.", "SENSITIVE"),
    ("Ich plane, nächsten Monat bei Siemens zu kündigen.", "SENSITIVE"),
    ("Das Sorgerecht für meine Tochter Lisa wird am Freitag verhandelt.", "SENSITIVE"),

    # SENSITIVE: Personal confessions
    ("Ich habe Suizidgedanken und weiß nicht, mit wem ich reden soll.", "SENSITIVE"),
    ("Ich bin schwanger und habe es meinem Arbeitgeber noch nicht gesagt.", "SENSITIVE"),
    ("Ich bin rückfällig geworden nach drei Jahren Abstinenz.", "SENSITIVE"),
    ("Ich habe meinen Partner betrogen und die Schuld frisst mich auf.", "SENSITIVE"),

    # SENSITIVE: Named relationships
    ("Meine Therapeutin Frau Weber meint, ich sollte öfter kommen.", "SENSITIVE"),
    ("Mein Nachbar Herr Schneider belästigt mich wegen der Grundstücksgrenze.", "SENSITIVE"),
    ("Mein Kollege Stefan hat mir erzählt, dass er den Chef wegen Betrug anzeigen will.", "SENSITIVE"),

    # PUBLIC: General knowledge
    ("Was ist die Hauptstadt von Australien?", "PUBLIC"),
    ("Erkläre den Unterschied zwischen Mitose und Meiose.", "PUBLIC"),
    ("Wie funktioniert Photosynthese?", "PUBLIC"),
    ("Was waren die Ursachen des Ersten Weltkriegs?", "PUBLIC"),
    ("Erkläre die Relativitätstheorie einfach.", "PUBLIC"),

    # PUBLIC: Programming/Technical
    ("Wie implementiere ich einen binären Suchbaum in Python?", "PUBLIC"),
    ("Was ist der Unterschied zwischen REST und GraphQL?", "PUBLIC"),
    ("Wie richte ich einen Kubernetes-Cluster ein?", "PUBLIC"),
    ("Erkläre das Observer-Pattern mit einem Codebeispiel.", "PUBLIC"),
    ("Wie funktioniert Garbage Collection in Go?", "PUBLIC"),
    ("Wie deploye ich eine FastAPI-Anwendung auf AWS Lambda?", "PUBLIC"),

    # PUBLIC: Generic health (no personal context)
    ("Was sind die Symptome von Typ-2-Diabetes?", "PUBLIC"),
    ("Wie wirkt Metformin im Körper?", "PUBLIC"),
    ("Was ist ein normaler Blutdruck?", "PUBLIC"),
    ("Welche Medikamente werden bei ADHS verschrieben?", "PUBLIC"),

    # PUBLIC: Generic legal/financial
    ("Wie funktioniert das Scheidungsrecht in Deutschland?", "PUBLIC"),
    ("Was passiert bei einer Privatinsolvenz?", "PUBLIC"),
    ("Wie hoch ist der Grundfreibetrag bei der Einkommensteuer 2026?", "PUBLIC"),
    ("Was ist der durchschnittliche Schufa-Score in Deutschland?", "PUBLIC"),
    ("Wie viel verdient ein Softwareentwickler in München?", "PUBLIC"),
    ("Was sind die Strafen für Trunkenheit am Steuer in Deutschland?", "PUBLIC"),

    # PUBLIC: Edge cases with names but no sensitive context
    ("Professor Müller hat einen guten Vortrag über Graphentheorie gehalten.", "PUBLIC"),
    ("Stefan und ich diskutieren, ob Python oder Rust besser für Systemprogrammierung ist.", "PUBLIC"),
    ("Dr. Schmidt hat ein tolles Paper über Transformer-Architekturen veröffentlicht.", "PUBLIC"),
    ("Meine Freundin Lisa hat mir dieses Rezept für Sauerteigbrot empfohlen.", "PUBLIC"),

    # SENSITIVE: Subtle personal context in German
    ("Welche Medikamente vertragen sich mit dem Sertralin, das ich gerade nehme?", "SENSITIVE"),
    ("Ich verdiene 4.500 Euro netto, reicht das für eine Wohnung in Frankfurt?", "SENSITIVE"),
    ("Seit meiner Scheidung fühle ich mich sehr einsam.", "SENSITIVE"),
    ("Mein Arzt hat bei der Untersuchung einen Knoten gefunden. Was sind die nächsten Schritte?", "SENSITIVE"),
    ("Ich bin auf Bewährung — darf ich ins Ausland reisen?", "SENSITIVE"),
]
