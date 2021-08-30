
_CATEGORIES_3D = [
    {'category': 'Children Cabinet'},
    {'category': 'Nightstand'},
    {'category': 'Bookcase / jewelry Armoire'},
    {'category': 'Wardrobe'},
    {'category': 'Coffee Table'},
    {'category': 'Corner/Side Table'},
    {'category': 'Sideboard / Side Cabinet / Console Table'},
    {'category': 'Wine Cabinet'},
    {'category': 'TV Stand'},
    {'category': 'Drawer Chest / Corner cabinet'},
    {'category': 'Shelf'},
    {'category': 'Round End Table'},
    {'category': 'King-size Bed'},
    {'category': 'Bunk Bed'},
    {'category': 'Bed Frame'},
    {'category': 'Single bed'},
    {'category': 'Kids Bed'},
    {'category': 'Dining Chair'},
    {'category': 'Lounge Chair / Cafe Chair / Office Chair'},
    {'category': 'Dressing Chair'},
    {'category': 'Classic Chinese Chair'},
    {'category': 'Barstool'},
    {'category': 'Dressing Table'},
    {'category': 'Dining Table'},
    {'category': 'Desk'},
    {'category': 'Three-Seat / Multi-seat Sofa'},
    {'category': 'armchair'},
    {'category': 'Loveseat Sofa'},
    {'category': 'L-shaped Sofa'},
    {'category': 'Lazy Sofa'},
    {'category': 'Chaise Longue Sofa'},
    {'category': 'Footstool / Sofastool / Bed End Stool / Stool'},
    {'category': 'Pendant Lamp'},
    {'category': 'Ceiling Lamp'}
]


import pickle
cat2id = {}
for i, v in enumerate(_CATEGORIES_3D):
    cat2id[v['category']] = i

with open('./cat2id_all.pkl', 'wb') as f:
    pickle.dump(cat2id, f)

