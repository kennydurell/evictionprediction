# Best sarimax - unemployment from six months ago and 1 year ago 1,1,1 2,1,1,6
# {'94102': 4.629852534758621, - better in baseline
#  '94103': 12.908029329353134, - better in baseline
#  '94107': 1.7060700864655898,
#  '94108': 3.6773549007211432,
#  '94109': 5.6109803064478987,
#  '94110': 8.1488491650299988,
#  '94112': 5.0505072127872399,
#  '94114': 3.4737614397836509, - better in baseline
#  '94115': 3.7055785521263043, - better in baseline
#  '94116': 3.1574215693037546, - better in baseline
#  '94117': 4.4724983928975606, - better in baseline
#  '94118': 3.4479037969697552, - better in baseline
#  '94121': 4.1315539075147028, - better in baseline
#  '94122': 3.8148970043365171, - better in baseline
#  '94123': 2.3038045507584464,
#  '94124': 3.1195089920430323,
#  '94127': 2.0205281308892693, - better in non-exog, check this
#  '94132': 33.605328065710154, - better in non-exog, check this
#  '94133': 4.9374601277839396, - better in baseline
#  '94134': 3.440297179625452} - better in non-exog, check this and better in baseline



top down hierarchical fill

{'94102': 5.9030161574577171,
 '94103': 18.352824934423317,
 '94104': 18.031105877298788,
 '94105': 1.0168494829582206,
 '94107': 1.9904053507525437,
 '94108': 2.2636880888325441,
 '94109': 5.2282786659108043,
 '94110': 6.8720631340232563,
 '94111': 1.032763167670834,
 '94112': 4.6762292438463984,
 '94114': 5.6699767850031524,
 '94115': 3.4798977670563369,
 '94116': 2.8224807490618327,
 '94117': 5.4757897638384163,
 '94118': 3.7826694421414984,
 '94121': 3.7902256104014391,
 '94122': 4.3848317900673752,
 '94123': 3.9359273418264382,
 '94124': 2.8060116694902377,
 '94127': 1.3274501578368112,
 '94129': 2.5532992463347268,
 '94130': 1.4175532589106703,
 '94131': 3.3323154634847025,
 '94132': 19.159411605030147,
 '94133': 4.1586923365101436,
 '94134': 3.3258180130646506,
 '94158': 0.36948386296262647,
 'Unknown_ZIP': 11.194824923560857}


ARIMA with white percentage added
 {'94102': [5.9858484260026392, 3],
  '94103': [17.397160749480967, 10],
  '94107': [1.5428431403625247, 3],
  '94108': [3.6845436046493831, 10],
  '94109': [6.5077617835614188, 3],
  '94110': [12.462489852345826, 4],
  '94111': [1.0, 3],
  '94112': [5.4504777498260895, 4],
  '94114': [4.5705459577006353, 6],
  '94115': [3.7547132567506316, 4],
  '94116': [3.1284941701858004, 4],
  '94117': [5.5971881523976981, 4],
  '94118': [3.8706020002436334, 4],
  '94121': [4.665569916587156, 4],
  '94122': [5.9716204812591709, 6],
  '94123': [2.2667022126737195, 4],
  '94124': [2.9219809605594933, 3],
  '94127': [2.0210481729620344, 3],
  '94131': [4.2349339077257815, 6],
  '94132': [6.7419899877366802, 3],
  '94133': [4.6823221581743679, 3],
  '94134': [3.7607480923964096, 4]}


ARIMA with tuned parameters and outliers being removed for each zip

{'94102': [1.9438647542611971, 1],
 '94103': [17.274552939685343, 10],
 '94107': [1.0, 1],
 '94108': [1.0989851924629965, 2],
 '94109': [4.8944432734887782, 3],
 '94110': [7.3392846792217643, 6],
 '94111': [0.4911636383018147, 3],
 '94112': [3.7260744957836214, 10],
 '94114': [3.1570731237444716, 3],
 '94115': [3.5989183145170962, 4],
 '94116': [1.4159978566064608, 2],
 '94117': [4.0642201665002311, 4],
 '94118': [2.518030506521181, 2],
 '94121': [3.6262501008494046, 3],
 '94122': [2.6479075053666321, 2],
 '94123': [2.0065768677375728, 4],
 '94124': [0.56666663514121374, 1],
 '94127': [1.4184304328797142, 3],
 '94131': [2.8835896749294969, 2],
 '94132': [14.052957482086613, 3],
 '94133': [4.1212120941600441, 2],
 '94134': [2.5552501370342573, 4]}

ARIMA with tuned parameters, no exog
{'94102': 5.3923949328749963,
 '94103': 27.04466775382172,
 '94105': 0.59033151748743062,
 '94107': 1.3862263238588584,
 '94108': 2.4396077886407692,
 '94109': 4.834006982881669,
 '94110': 6.6418385726737696,
 '94112': 4.1577858584578689,
 '94114': 3.2927207410350459,
 '94115': 3.2428119105852309,
 '94116': 2.5986613498696074,
 '94117': 3.9900053899491135,
 '94118': 3.3731751285164435,
 '94121': 3.2976466304568675,
 '94122': 3.5821570849976032,
 '94123': 2.1804600216537944,
 '94124': 2.6767481389520507,
 '94127': 1.3775557144783099,
 '94131': 3.0153807166997963,
 '94132': 26.870496902837179,
 '94133': 4.2320881454047496,
 '94134': 2.6254756580349299}

Baseline
{'94131': 3.5289390635985303, '94132': 34.507890594511537, '94133': 4.0279766448865892, '94134': 2.9775720242488299, '94118': 2.8598158982284434, '94112': 6.2091618850713148, '94110': 12.219941028719537, '94111': 4.9986160087429798, '94116': 2.9843049176424548, '94117': 4.4374911254601104, '94114': 3.2429659920396428, '94115': 3.1160709633931734, '94127': 4.5547224500157508, '94124': 3.376353381819718, '94123': 2.6887325174740302, '94122': 3.4219858377305261, '94121': 3.3891749579288293, '94109': 7.3678926942754908, '94108': 4.8969655420079237, '94103': 11.567840937727876, '94102': 4.3320396590943258, '94105': 5.0553822563860527, '94104': 5.5258327396411087, '94107': 4.0869536499420303}



Best forest with mean imputed for training data greater than 3 STD from mean of data
{'94109': 5.1817845915419332, '94108': 4.2856271513789643, '94118': 2.8850889172332441, '94112': 4.9956968737576393, '94110': 7.9629411448777478, '94102': 4.117892184685858, '94116': 3.1842900405298247, '94117': 4.3614165699944287, '94114': 3.9812131280985112, '94115': 3.0341055832949158, '94127': 2.2475012709874544, '94131': 2.6435986086282637, '94132': 35.371336635616672, '94124': 3.166703621629281, '94123': 2.4433383638965891, '94107': 2.5034631350806755, '94121': 3.4578364385051921, '94134': 2.478532098261105, '94103': 11.801382355675171, '94133': 4.6551793954859679, '94122': 3.3049937114575521}



2nd Best Forest
{'94131': 2.7320765773020392, '94132': 34.952857139613386, '94133': 4.8168622735305773, '94134': 2.8900815608856836, '94118': 3.3600181877814626, '94112': 5.0070220960437419, '94110': 8.2202123994390739, '94111': 0.74274266517190646, '94116': 3.9255572852781042, '94117': 5.3762239335600395, '94114': 3.9242833740697165, '94115': 3.7112815995436281, '94127': 1.9988039902162362, '94124': 2.9140464420915917, '94123': 3.9729490714486038, '94122': 3.309078421554859, '94121': 4.0643805418771004, '94109': 4.6317062409767198, '94108': 3.954790812466618, '94103': 14.427563709600959, '94102': 5.3254615331011959, '94105': 1.1803954139750517, '94104': 0.5, '94107': 2.1746407749327243}

#Best sarimax with no exogenous - all 1,1,1 1,1,1,12 for p,d,q P,D,Q,S
#{'94102': 7.0664784226736259,
#  '94103': 32.864507696727912,
#  '94107': 1.8762496148670962,
#  '94109': 6.3122197310239976,
#  '94110': 8.3078312816425601,
#  '94112': 6.4814593173184472,
#  '94114': 3.6817356832284047,
#  '94115': 4.301800870693901,
#  '94116': 3.680716958275188,
#  '94117': 4.744228061940329,
#  '94118': 4.1339630928517694,
#  '94121': 4.4055544523536536,
#  '94122': 4.5135517079430922,
#  '94123': 2.8087427577934601,
#  '94124': 3.3589269750048696,
#  '94127': 1.8452635062796088,
#  '94131': 3.6671607031582756,
#  '94132': 28.390857910814866,
#  '94133': 5.5495515052501352,
#  '94134': 3.3825218654015892}

2nd best sarimax - 1,1,2 2,1,1,6 - best 94103 found, slightly higher for all others
than best

{'94102': 4.5330251517929945,
 '94103': 11.9059030908554,
 '94107': 1.8562151487363068,
 '94108': 3.8426491449818574,
 '94109': 6.0586341422112637,
 '94110': 8.57025129827373,
 '94112': 5.1416049896927287,
 '94114': 3.6005061993692711,
 '94115': 3.9024190003923764,
 '94116': 3.1300540016730927,
 '94117': 4.6220727216520032,
 '94118': 3.6655534461215225,
 '94121': 4.0802517528757107,
 '94122': 3.9576487066712676,
 '94123': 2.3430366302209564,
 '94124': 3.0628482508219403,
 '94127': 2.1602599296808696,
 '94132': 34.30139665083351,
 '94133': 5.1268026102714908,
 Figures'94134': 3.7552143790713099}

#2nd Best sarimax - includes unemployment from previous year
# {'94102': 4.5372978592954416,
#  '94103': 15.953056102179051,
#  '94107': 1.756172470924739,
#  '94108': 3.3189031205861892,
#  '94109': 6.6426772867611641,
#  '94110': 8.605328289078118,
#  '94112': 6.127551560563659,
#  '94114': 3.5327083608235905,
#  '94115': 3.9181131092519577,
#  '94116': 3.5115326838718048,
#  '94117': 4.205555339306402,
#  '94118': 3.6875245429602646,
#  '94121': 4.5009196764944202,
#  '94122': 4.386821011210376,
#  '94123': 2.3355447827512918,
#  '94124': 2.9547316001907076,
#  '94127': 1.9805148125986205,
#  '94132': 30.181500966433024,
#  '94133': 5.5563353133133688,
#  '94134': 3.3832831743361393}



{'94102': [(2, 1, 1), 532.5856587200159],
'94103': [(0, 1, 1), 803.3662428898983],
'94104': ['order', 40000000000],
'94105': [(5, 0, 0), 13.658867630369574],
'94107': [(0, 0, 0), 271.72564884092628],
'94108': [(0, 0, 0), 309.45002369236113],
'94109': [(1, 1, 1), 513.2303675590061],
'94110': [(4, 1, 1), 558.9043335374527],
'94111': [(0, 0, 0), 59.731675440435893],
'94112': [(1, 1, 1), 489.46407667582207],
'94114': [(0, 1, 1), 437.10399266503805],
'94115': [(1, 1, 1), 425.69702383134984],
'94116': [(0, 1, 1), 387.8427126258089],
'94117': [(0, 1, 1), 484.74695270219246],
'94118': [(0, 1, 1), 424.40096173225675],
'94121': [(0, 1, 1), 442.91657962821625],
'94122': [(2, 1, 1), 451.9743459843794],
'94123': [(0, 0, 0), 385.11084487071133],
'94124': [(0, 1, 1), 404.7179036313577],
'94127': [(0, 0, 0), 193.42690896197888],
'94131': [(2, 0, 1), 353.0419882704688],
'94132': [(0, 1, 1), 768.1951226809797],
'94133': [(0, 1, 1), 475.35291292581576],
'94134': [(3, 1, 1), 363.41584411489987]}


{'94102': [40000000, 'none'],
'94103': [40000000, 'none'],
'94107': [40000000, 'none'],
'94108': [40000000, 'none'],
'94109': [40000000, 'none'],
'94110': [40000000, 'none'],
'94111': [40000000, 'none'],
'94112': [40000000, 'none'],
'94114': [40000000, 'none'],
'94115': [40000000, 'none'],
'94116': [40000000, 'none'],
'94117': [40000000, 'none'],
'94118': [40000000, 'none'],
'94121': [40000000, 'none'],
'94122': [40000000, 'none'],
'94123': [40000000, 'none'],
'94124': [40000000, 'none'],
'94127': [40000000, 'none'],
'94131': [40000000, 'none'],
'94132': [40000000, 'none'],
'94133': [40000000, 'none'],
'94134': [40000000, 'none']}


Random Forest - white percentage, black population previous year, previous two years
{'94102': (4.7992429881105805, 0.46117363150127932, [10, 'auto', 3]),
 '94103': (14.054667936956758, 2.4799093710233322, [10, 'auto', 3]),
 '94104': (1.1709419285344598, -3.6700027061849587, [1000, 'auto', 10]),
 '94105': (1.1503622617824931, -3.8910384500385113, [10, 'auto', 3]),
 '94107': (1.7695468063885735, -2.2995515728822289, [1000, 'auto', 10]),
 '94108': (4.5643701478028538, -0.3169910656686028, [1000, 'auto', 10]),
 '94109': (5.2166594401288755, -2.166231258412747, [1000, 'auto', 10]),
 '94110': (8.1482206646604762, -4.0904642877770208, [10, 'auto', 3]),
 '94111': (2.6143832924802743, -2.3607686211505707, [10, 'auto', 3]),
 '94112': (4.6371342504981303, -1.5869402857596482, [1000, 'auto', 10]),
 '94114': (3.6601135893412469, 0.4119972020593301, [10, 'auto', 3]),
 '94115': (3.8673045902951126, 0.75176917742493865, [1000, 'auto', 10]),
 '94116': (3.402539979821797, 0.4248807698042647, [1000, 'auto', 10]),
 '94117': (5.1290827361513402, 0.68680527413393566, [1000, 'auto', 10]),
 '94118': (3.3934751578667588, 0.53508781021341889, [1000, 'auto', 10]),
 '94121': (3.621516197869143, 0.22637346628626176, [1000, 'auto', 10]),
 '94122': (3.4946662501589474, 0.05820281505304381, [1000, 'auto', 10]),
 '94123': (2.5658749551983795, -0.11465337387501373, [10, 'auto', 3]),
 '94124': (3.0139382092889306, -0.35000798297255553, [10, 'auto', 3]),
 '94127': (2.2669650272517679, -2.2714294385958249, [10, 'auto', 3]),
 '94131': (2.2530265021421383, -1.2604919546921654, [1000, 'auto', 10]),
 '94132': (35.565606339056075, 1.0559975267095822, [1000, 'auto', 10]),
 '94133': (4.285657142476186, 0.24612855856050775, [10, 'auto', 3]),
 '94134': (2.983759977957444, 0.020749079595408393, [1000, 'auto', 10])}