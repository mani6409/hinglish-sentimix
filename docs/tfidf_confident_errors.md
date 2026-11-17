# TF-IDF word 1-2: the most confident mistakes on dev

Reproduce with `python -m src.tfidf_error_analysis`. Model fitted on train only; these are dev tweets.

- Dev errors: 1164 of 3000. Of these, 537 (46%) have the true label neutral.
- Among the 30 errors the model was most sure about, 26 have the true label neutral.
- Read them yourself: do you think the gold label is wrong, or is the model wrong?

| uid | true | predicted | confidence | tweet | my verdict (label wrong / model wrong / unclear) |
|---|---|---|---|---|---|
| 38861 | neutral | positive | 0.94 | @ Zinda _ Laash _ Hon Happy birthday to you . many many happy returns of the day . May Allah bless you .🎂🎈🎉🎇🎆🎊🎗🎁🎀 | |
| 17789 | neutral | positive | 0.94 | Sweet 😍😘 Good morning MSG Papa ji Bless me pita ji sewa simran parmarth and good health 🙏🙏 # MondayMorning https // t co / D6LUkgmxRX | |
| 11679 | neutral | positive | 0.94 | Wish him a very happy birthday ! Love !❤️🌹 https // t . co / LGeTWEgi8S | |
| 44393 | neutral | positive | 0.93 | Love you har https // t . co / X61UykGZZ6 | |
| 7044 | neutral | positive | 0.92 | Hii i love u sunshine ❤❤❤❤❤ — Hiii I love you more sweetie ❤❤❤👑 https // t . co / o2reBKOa84 | |
| 36789 | neutral | positive | 0.92 | @ vijaykbandhu Many more days and years of good health good life and prosperity Happy Birthday . | |
| 20017 | neutral | negative | 0.90 | @ rizwanqasmi7 @ GhulamJ64765462 @ FiyazChohanPTI Shame on you ksi ki maa ko q ghest rhy gandi or ghtia politics main | |
| 42337 | neutral | negative | 0.90 | Baimaan tera Ajay Singh Cheating baaj Teri Airlines .. Saale Chutiya banane ke liye Baithe ho .. Jakar Station me Bhi … https // t . co / BG2HpIjJ9j | |
| 27476 | neutral | positive | 0.90 | @ Nikita12S Mere taraf se aapki mom Ko unki Janam diwas ki dher sari subhkamnaye 🙏 https // t co / fM9bigX583 | |
| 26705 | neutral | positive | 0.90 | Women Power Smriti Irani Ko Dher Saari Subhkamnaye We all proud of u mam # HarGharModi ✌️✌️✌️✌️ @ smritiirani @ BJP4India @ narendramodi | |
| 1338 | neutral | positive | 0.88 | RT @ JunaNJoey Thank you @ CelebMix for such an awesome interview Thank you @ Lauraklonowski whooo hoooo https // t co / 2sbJgiunov | |
| 27969 | neutral | negative | 0.87 | Ye kya bc Yahan bjp jeeti hai Jo result aaya hai wahi sahi rhe Pr sala ye news fake ho warna election pr se bh … https // t co / PcImaCay21 | |
| 1400 | neutral | negative | 0.87 | @ SAMBIT _ PATRAg @ anjanaomkashyap Sab she pahele tere gahr me kanon lago kr kiyo ke teri faimly pr tera kabo nahi he … https // t . co / scfwME6S7m | |
| 17784 | neutral | positive | 0.87 | @ GuruOfficial All the best Guru we know u rock the live show we always love u Kabhi hamare city mein bhi live … https // t co / M1nJUTfV0w | |
| 28368 | neutral | positive | 0.86 | RT @ ForeverRed98 YOU COME AT THE KING YOU BEST NOT MISS !!!! https // t . co / d6hEQRUtGu | |
| 7702 | neutral | positive | 0.86 | @ narendramodi Shri Narendra Modi Ji pranam mai aap ko dekh kar bahut prabhavit hua hun mera man aap se milne aur aa … https // t co / OBcV9L59YC | |
| 10866 | neutral | positive | 0.86 | RT @ mohitsaini7915 # OneSolutionToEveryProblem Gud morning Sai @ Gurmeetramrahim ji Bless everyone with maximum sewa Simran Parmarth And S … | |
| 4396 | negative | positive | 0.86 | Allah .. My Lord forgive us . Receive us our repentance . Bless us on this beautiful month Accept our fast and ibadat .. ❤ | |
| 18321 | neutral | positive | 0.86 | RT @ sohaib _ khan7 Happy New Year O ho Sorry 🤪🤪 Good Morning Shit 🤭🤭 Happy Birthday Uff 😔😔 I Want 2 Say Best Wishes in Ur Wedd … | |
| 43688 | neutral | positive | 0.85 | @ narendramodi congratulations and best of luck sir for your second term Bajrang Bali aako Shakti de | |
| 3639 | negative | positive | 0.85 | Happy Birthday @ aadiivaasii May you live like a normal Indian citizen soon . Wish you come of the tribes soon . Ab … https // t . co / edavtS9qKP | |
| 31839 | neutral | negative | 0.84 | Kagiso Rabada Virat Kohli is immature can't take abuse . Indian cricket fans BC MC tu hoga immature . Saale … https // t . co / j4ZPtzYLXK | |
| 10966 | positive | negative | 0.84 | RT @ Himansh19149288 @ Anastaciuss @ Vyas86Manish @ shahnaveid Soldier ni chutiya aatankwadi mare hai . Teri tuche desh ki aukat hai to ukhad l … | |
| 11143 | neutral | negative | 0.84 | @ Aakash _ SRKian Hum log teachers ko pagal bana ke class se nikalte the ye log maa baap ka chutiya bana karke date pe nikalte hai 😂😭 | |
| 13731 | positive | negative | 0.83 | @ Hussain _ NSharif @ MaryamNSharif To BC tu or Mera baap loot k kha Gaye hain . Un paiso main se he sahe in k help kar de . Bublo k bacha | |
| 25858 | neutral | positive | 0.83 | RT @ fatma _ karume Sina CHAMA CHOCHOTE ! I am a TZnian first . It makes life so much easier . Maana I am not blinded by party politics . Imagine … | |
| 8692 | neutral | negative | 0.83 | @ aajtak logon ko kam chor bana rahe hain ye log hamari mahnat ka paisa hai aise nahi udaya karte apne ghar k pai … https // t co / yD0w70nWp1 | |
| 23256 | neutral | positive | 0.83 | @ BeingSalmanKhan God bless you salman ji app best or lucky person ho app ki sari Umar tarki ho God bless you | |
| 2281 | neutral | positive | 0.82 | @ MaryamNSharif The day of my success salute Mian sahib you really deserve fr love n respect . Whole nation is wait … https // t . co / vCmngCouBX | |
| 19658 | neutral | positive | 0.82 | @ Aadii _ Sawant Dynamo I'm a huge fan of you 😊 Keep growing ... (｡♥‿♥｡) We all support n love you ◉‿◉ 🤩 kabhi waqt mile … https // t . co / u1VAqvLmP8 | |
