TEST_CASES = [
    # =========================================================================
    # SENSITIVE: Personal Identifiers (SSN, DOB, passport, driver's license)
    # =========================================================================
    ("My SSN is 478-33-1234, can you help me fill out this tax form?", "SENSITIVE"),
    ("I was born on March 14, 1987 and need to verify my identity.", "SENSITIVE"),
    ("My passport number is X12345678, is it valid for travel to Japan?", "SENSITIVE"),
    ("Can you look up my driver's license? It's DL-9928374-CA.", "SENSITIVE"),
    ("My date of birth is 09/22/1995 and I need to update my insurance.", "SENSITIVE"),
    ("I need help—my social security number 321-45-6789 may have been compromised.", "SENSITIVE"),
    ("My national ID number is 8501015800085, how do I renew it?", "SENSITIVE"),
    ("Please verify my passport: US passport number 987654321, issued 2019.", "SENSITIVE"),

    # =========================================================================
    # SENSITIVE: Contact Info (email, phone, physical address)
    # =========================================================================
    ("My email is jane.smith92@gmail.com, can you send the report there?", "SENSITIVE"),
    ("Reach me at 415-555-0173 after 5pm to discuss the results.", "SENSITIVE"),
    ("I live at 742 Evergreen Terrace, Springfield, IL 62704.", "SENSITIVE"),
    ("My new phone number is +44 7911 123456, please update my records.", "SENSITIVE"),
    ("Send the package to 1600 Pennsylvania Ave NW, Washington DC 20500, attention Sarah.", "SENSITIVE"),
    ("You can contact my wife at maria.gonzalez@outlook.com.", "SENSITIVE"),
    ("My home address is Apt 4B, 221B Baker Street, London NW1 6XE.", "SENSITIVE"),

    # =========================================================================
    # SENSITIVE: Financial (salary, bank accounts, credit cards, debts, credit score)
    # =========================================================================
    ("My annual salary is $127,000 and I'm wondering if I should negotiate.", "SENSITIVE"),
    ("My bank account number is 0012345678 at Chase, routing 021000021.", "SENSITIVE"),
    ("My credit card number is 4532-1234-5678-9012, exp 08/27, CVV 331.", "SENSITIVE"),
    ("I owe $45,000 in student loans and $12,000 on credit cards.", "SENSITIVE"),
    ("My credit score dropped to 580 after missing two mortgage payments.", "SENSITIVE"),
    ("I make $8,500/month but my rent is $3,200. Can I afford a new car?", "SENSITIVE"),
    ("My 401k balance is $234,000 and I'm thinking about early withdrawal.", "SENSITIVE"),
    ("I have $15,000 in savings and $78,000 remaining on my mortgage.", "SENSITIVE"),
    ("My wife and I have a combined household income of $195,000.", "SENSITIVE"),
    ("I just received an inheritance of $500,000 from my grandmother's estate.", "SENSITIVE"),

    # =========================================================================
    # SENSITIVE: Health/Medical (diagnoses, medications, conditions, doctor visits)
    # =========================================================================
    ("I was diagnosed with Type 2 diabetes last month and I'm scared.", "SENSITIVE"),
    ("My doctor prescribed me 20mg of Lexapro for my anxiety disorder.", "SENSITIVE"),
    ("I have a colonoscopy scheduled for next Tuesday at Mount Sinai.", "SENSITIVE"),
    ("I've been taking metformin 500mg twice daily but my blood sugar is still high.", "SENSITIVE"),
    ("My therapist says I might have PTSD from the accident last year.", "SENSITIVE"),
    ("I tested positive for HIV and don't know how to tell my partner.", "SENSITIVE"),
    ("My son was diagnosed with ADHD and the school wants an IEP.", "SENSITIVE"),
    ("I had a miscarriage at 11 weeks and I'm devastated.", "SENSITIVE"),
    ("My blood pressure reading was 180/110 at my last checkup with Dr. Patel.", "SENSITIVE"),
    ("I'm on 40mg of Adderall and worried about dependency.", "SENSITIVE"),

    # =========================================================================
    # SENSITIVE: Legal/Private (arrests, lawsuits, divorce, custody, resignation)
    # =========================================================================
    ("I was arrested for DUI last weekend and need a lawyer.", "SENSITIVE"),
    ("My ex-wife is suing me for full custody of our two children.", "SENSITIVE"),
    ("I'm filing for divorce after discovering my husband's affair.", "SENSITIVE"),
    ("I was fired for alleged misconduct but I believe it was retaliation.", "SENSITIVE"),
    ("I'm planning to resign from my position at Google next month.", "SENSITIVE"),
    ("I received a cease and desist letter from my former employer.", "SENSITIVE"),
    ("I'm being investigated by the IRS for unreported income.", "SENSITIVE"),
    ("My landlord is evicting me illegally and I need to know my rights.", "SENSITIVE"),
    ("I was involved in a car accident and the other driver is threatening to sue.", "SENSITIVE"),

    # =========================================================================
    # SENSITIVE: Named Personal Relationships
    # =========================================================================
    ("My boss Mike keeps making inappropriate comments at work.", "SENSITIVE"),
    ("My doctor, Dr. Sarah Chen, recommended I get a second opinion.", "SENSITIVE"),
    ("My therapist Janet thinks I should increase my session frequency.", "SENSITIVE"),
    ("My coworker Dave told me he's planning to report our manager for fraud.", "SENSITIVE"),
    ("My neighbor Tom has been harassing me about the property line.", "SENSITIVE"),
    ("My ex-girlfriend Rachel keeps calling me despite the restraining order.", "SENSITIVE"),
    ("My son's teacher Mrs. Williams says he's been acting out in class.", "SENSITIVE"),

    # =========================================================================
    # SENSITIVE: Emotional/Intimate Confessions
    # =========================================================================
    ("I've been having suicidal thoughts and I don't know who to talk to.", "SENSITIVE"),
    ("I think I'm falling in love with my married coworker.", "SENSITIVE"),
    ("I haven't told anyone but I've been struggling with bulimia for years.", "SENSITIVE"),
    ("I'm deeply ashamed—I relapsed on alcohol after being sober for 3 years.", "SENSITIVE"),
    ("I cheated on my partner and the guilt is consuming me.", "SENSITIVE"),
    ("I feel like a terrible mother because I sometimes resent my children.", "SENSITIVE"),
    ("I was sexually assaulted in college and never reported it.", "SENSITIVE"),
    ("I've been secretly gambling online and lost over $20,000.", "SENSITIVE"),

    # =========================================================================
    # PUBLIC: General Knowledge Questions
    # =========================================================================
    ("What is the capital of Australia?", "PUBLIC"),
    ("How does photosynthesis work?", "PUBLIC"),
    ("What caused the fall of the Roman Empire?", "PUBLIC"),
    ("Explain the difference between mitosis and meiosis.", "PUBLIC"),
    ("What is the speed of light in a vacuum?", "PUBLIC"),
    ("Who wrote 'One Hundred Years of Solitude'?", "PUBLIC"),
    ("How many planets are in the solar system?", "PUBLIC"),
    ("What is the Pythagorean theorem?", "PUBLIC"),
    ("When did World War II end?", "PUBLIC"),
    ("What is the largest ocean on Earth?", "PUBLIC"),

    # =========================================================================
    # PUBLIC: Programming/Coding Questions
    # =========================================================================
    ("How do I reverse a linked list in Python?", "PUBLIC"),
    ("What's the difference between REST and GraphQL?", "PUBLIC"),
    ("Explain the concept of dependency injection.", "PUBLIC"),
    ("How do I set up a PostgreSQL database with Docker?", "PUBLIC"),
    ("What is the time complexity of quicksort?", "PUBLIC"),
    ("Write a function to check if a string is a palindrome.", "PUBLIC"),
    ("How do I implement a binary search tree in Java?", "PUBLIC"),
    ("What's the difference between TCP and UDP?", "PUBLIC"),
    ("Explain how garbage collection works in Go.", "PUBLIC"),
    ("How do I deploy a Flask app to AWS Lambda?", "PUBLIC"),

    # =========================================================================
    # PUBLIC: Generic Health/Science (no personal context)
    # =========================================================================
    ("What are the symptoms of Type 2 diabetes?", "PUBLIC"),
    ("How does the immune system fight viruses?", "PUBLIC"),
    ("What is the recommended daily intake of vitamin D?", "PUBLIC"),
    ("How do mRNA vaccines work?", "PUBLIC"),
    ("What are the stages of human embryonic development?", "PUBLIC"),
    ("Explain the difference between bacteria and viruses.", "PUBLIC"),
    ("What causes Alzheimer's disease?", "PUBLIC"),
    ("How does general anesthesia affect the brain?", "PUBLIC"),

    # =========================================================================
    # PUBLIC: Research/Tutorial Requests
    # =========================================================================
    ("Write a tutorial on building a REST API with FastAPI.", "PUBLIC"),
    ("Summarize the key ideas of Kahneman's 'Thinking, Fast and Slow'.", "PUBLIC"),
    ("Explain the basics of quantum computing for beginners.", "PUBLIC"),
    ("What are best practices for database indexing?", "PUBLIC"),
    ("Give me an overview of reinforcement learning algorithms.", "PUBLIC"),
    ("How do transformers work in natural language processing?", "PUBLIC"),
    ("Write a beginner's guide to Kubernetes.", "PUBLIC"),
    ("Explain the CAP theorem in distributed systems.", "PUBLIC"),

    # =========================================================================
    # PUBLIC: Questions ABOUT PII Concepts (not containing actual PII)
    # =========================================================================
    ("What is personally identifiable information under GDPR?", "PUBLIC"),
    ("How should companies handle data subject access requests?", "PUBLIC"),
    ("What are the penalties for HIPAA violations?", "PUBLIC"),
    ("Explain the difference between PII and PHI.", "PUBLIC"),
    ("What data anonymization techniques are most effective?", "PUBLIC"),
    ("How does differential privacy work?", "PUBLIC"),
    ("What are the CCPA requirements for data deletion?", "PUBLIC"),
    ("How do you implement data masking in a production database?", "PUBLIC"),

    # =========================================================================
    # TRICKY EDGE CASES: Code Examples Containing Fake PII Data
    # =========================================================================
    ("Here's my test data: user = {'name': 'John Doe', 'ssn': '123-45-6789', 'email': 'john@example.com'}. How do I validate this schema?", "PUBLIC"),
    ("I'm writing unit tests and need to generate fake credit card numbers like 4111-1111-1111-1111. What library should I use?", "PUBLIC"),
    ("My regex for phone numbers r'\\d{3}-\\d{3}-\\d{4}' isn't matching +1 (555) 123-4567. How do I fix it?", "PUBLIC"),
    ("I have a CSV with columns: name, email, phone, ssn. How do I write a PII scanner to detect these fields?", "PUBLIC"),
    ("In my test fixture I use address='123 Main St, Anytown USA 12345'. Is there a better fake data generator?", "PUBLIC"),
    ("How do I mask fields like 'password' and 'credit_card' in my API logs? Example payload: {password: 'hunter2', cc: '4532-xxxx-xxxx-1234'}.", "PUBLIC"),
    ("I need a function that validates SSN format (XXX-XX-XXXX) without storing the actual number. Can you write one?", "PUBLIC"),
    ("My Faker library generates data like Name: Jane Smith, DOB: 1990-01-15. How do I seed it for reproducibility?", "PUBLIC"),

    # =========================================================================
    # TRICKY EDGE CASES: Generic Questions About Sensitive Topics
    # =========================================================================
    ("What are the grounds for divorce in California?", "PUBLIC"),
    ("How do federal tax brackets work for 2026?", "PUBLIC"),
    ("What happens during a bankruptcy filing?", "PUBLIC"),
    ("What are the legal requirements for a restraining order?", "PUBLIC"),
    ("How does child custody work in joint custody arrangements?", "PUBLIC"),
    ("What is the process for filing a wrongful termination claim?", "PUBLIC"),
    ("What are the penalties for a first-time DUI offense?", "PUBLIC"),
    ("How do plea bargains work in the criminal justice system?", "PUBLIC"),

    # =========================================================================
    # TRICKY EDGE CASES: Messages With Names but No Sensitive Context
    # =========================================================================
    ("Can you recommend books similar to what Sarah reads—mostly sci-fi?", "PUBLIC"),
    ("Mike and I are debating whether Python or Rust is better for systems programming.", "PUBLIC"),
    ("Dr. Smith published a great paper on transformer architectures.", "PUBLIC"),
    ("I'm meeting Dave for coffee to discuss the open source project.", "PUBLIC"),
    ("Professor Johnson's lecture on graph theory was really helpful.", "PUBLIC"),
    ("My friend Lisa recommended this recipe for sourdough bread.", "PUBLIC"),

    # =========================================================================
    # TRICKY EDGE CASES: Medical/Legal Questions Without Personal Framing
    # =========================================================================
    ("What medications are commonly prescribed for ADHD?", "PUBLIC"),
    ("What is the success rate of knee replacement surgery?", "PUBLIC"),
    ("How do SSRIs compare to SNRIs for treating depression?", "PUBLIC"),
    ("What are the side effects of metformin?", "PUBLIC"),
    ("What is the statute of limitations for personal injury claims?", "PUBLIC"),
    ("How does expungement of criminal records work?", "PUBLIC"),

    # =========================================================================
    # TRICKY EDGE CASES: Ambiguous Salary/Financial Questions (no personal data)
    # =========================================================================
    ("What's a good salary for a senior software developer in NYC?", "PUBLIC"),
    ("What is the average credit score in the United States?", "PUBLIC"),
    ("How much should I have in my emergency fund as a general rule?", "PUBLIC"),
    ("What are typical interest rates for a 30-year fixed mortgage?", "PUBLIC"),
    ("What is the median household income in Texas?", "PUBLIC"),
    ("How much do teachers typically earn in Germany?", "PUBLIC"),

    # =========================================================================
    # TRICKY EDGE CASES: Subtle Personal Context That Makes It Sensitive
    # =========================================================================
    ("What medications interact with the Zoloft I'm currently taking?", "SENSITIVE"),
    ("I earn $85k—is that enough to qualify for a mortgage in Denver?", "SENSITIVE"),
    ("My therapist recommended CBT. What should I expect in my first session?", "SENSITIVE"),
    ("I was denied a loan because of my credit history. What can I do?", "SENSITIVE"),
    ("After my DUI last year, how long until I can get my license back?", "SENSITIVE"),
    ("Since my divorce, I've been struggling with loneliness.", "SENSITIVE"),
    ("I'm on probation—can I travel out of state for a conference?", "SENSITIVE"),
    ("My doctor found a lump during my exam. What are the next steps?", "SENSITIVE"),

    # =========================================================================
    # TRICKY EDGE CASES: Questions That Sound Sensitive but Are Generic
    # =========================================================================
    ("How do I change my name legally after marriage?", "PUBLIC"),
    ("What documents do you need to apply for a passport?", "PUBLIC"),
    ("What is the process for adopting a child internationally?", "PUBLIC"),
    ("How do life insurance payouts work after someone dies?", "PUBLIC"),
    ("What are the steps to apply for disability benefits?", "PUBLIC"),
    ("How do you report identity theft to the FTC?", "PUBLIC"),

    # =========================================================================
    # TRICKY EDGE CASES: Mixed / Borderline Cases
    # =========================================================================
    ("Write an email template for submitting a resignation letter.", "PUBLIC"),
    ("I need to write a complaint about my manager—what should I include?", "SENSITIVE"),
    ("How do I write a letter to dispute a charge on my credit card statement for $847?", "SENSITIVE"),
    ("What's the best way to tell my kids about the divorce?", "SENSITIVE"),
    ("Help me draft a message to my landlord about the mold problem at my apartment on 5th Avenue.", "SENSITIVE"),
    ("What questions should I ask during a job interview at a startup?", "PUBLIC"),
    ("I need to appeal my health insurance claim denial for my surgery.", "SENSITIVE"),
    ("Explain how to calculate capital gains tax on stock sales.", "PUBLIC"),
    ("I sold 500 shares of AAPL at $190 and need to calculate my tax liability.", "SENSITIVE"),
    ("What is a good strategy for paying off $30,000 in debt?", "PUBLIC"),

    # =========================================================================
    # TRICKY EDGE CASES: Requests Mentioning PII Terms in Abstract/Educational Context
    # =========================================================================
    ("Write a Python class that detects email addresses and phone numbers in text.", "PUBLIC"),
    ("What regex pattern matches US Social Security numbers?", "PUBLIC"),
    ("How should a HIPAA-compliant system store patient records?", "PUBLIC"),
    ("Design a database schema for storing encrypted credit card information.", "PUBLIC"),
    ("What are the NIST guidelines for protecting PII?", "PUBLIC"),

    # =========================================================================
    # ADDITIONAL SENSITIVE: Workplace / HR
    # =========================================================================
    ("I'm being sexually harassed by my supervisor and HR won't do anything.", "SENSITIVE"),
    ("I discovered my company is committing tax fraud. Should I whistleblow?", "SENSITIVE"),
    ("My performance review was unfair—my manager rated me low because of my pregnancy.", "SENSITIVE"),

    # =========================================================================
    # ADDITIONAL SENSITIVE: Immigration / Identity
    # =========================================================================
    ("My visa expires next month and my green card application is still pending.", "SENSITIVE"),
    ("I'm undocumented and afraid to seek medical care. What are my options?", "SENSITIVE"),
    ("My asylum application was denied and I'm facing deportation.", "SENSITIVE"),

    # =========================================================================
    # ADDITIONAL SENSITIVE: Personal Situations / Edge Cases
    # =========================================================================
    ("I'm pregnant and haven't told my employer yet. When am I legally required to?", "SENSITIVE"),
    ("My teenager was caught shoplifting and has a court date next week.", "SENSITIVE"),
    ("I just found out my husband has a secret bank account.", "SENSITIVE"),
    ("My elderly mother has dementia and I need to get power of attorney.", "SENSITIVE"),
    ("I failed a drug test at work and might get fired.", "SENSITIVE"),
    ("I'm $3,000 behind on child support payments. What are the consequences?", "SENSITIVE"),
    ("My identity was stolen and someone opened credit cards in my name.", "SENSITIVE"),
    ("I have a benign tumor that my insurance won't cover removal of.", "SENSITIVE"),
    ("I was passed over for promotion because of my age—I'm 58.", "SENSITIVE"),
    ("My ex is posting intimate photos of me online. How do I get them removed?", "SENSITIVE"),
    ("I lied on my resume about having a degree and now they're doing a background check.", "SENSITIVE"),
    ("My brother was just diagnosed with schizophrenia and the family doesn't know what to do.", "SENSITIVE"),
    ("I owe the IRS $23,000 in back taxes from my freelance work.", "SENSITIVE"),
    ("I'm addicted to painkillers that were originally prescribed after my surgery.", "SENSITIVE"),
    ("My spouse and I are in couples therapy because of trust issues.", "SENSITIVE"),
    ("I think my accountant has been embezzling from my business.", "SENSITIVE"),
    ("My daughter's school called—she told her teacher she wants to hurt herself.", "SENSITIVE"),
    ("I got a letter saying my wages will be garnished starting next month.", "SENSITIVE"),
    ("I was in a psych ward last year and worried it will show up on a background check.", "SENSITIVE"),
]
