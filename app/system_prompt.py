from typing import Dict, List

_STATUS_LABELS: Dict[str, str] = {
    "pending": "na čekanju",
    "processing": "u obradi",
    "shipped": "poslata",
    "delivered": "isporučena",
    "cancelled": "otkazana",
    "on-hold": "pauzirana",
    "completed": "završena",
    "refunded": "refundirana",
    "failed": "neuspešna",
}


def _localize_status(status: str) -> str:
    return _STATUS_LABELS.get((status or "").lower(), status or "nepoznat")


def format_product_card(products: List[Dict]) -> str:
    if not products:
        return "Nisu pronađeni proizvodi koji odgovaraju upitu."
    lines = ["Pronađeni proizvodi:"]
    for p in products:
        name = p.get("name") or p.get("short_name", "Nepoznat")
        price = p.get("price", "")
        sale_price = p.get("sale_price")
        in_stock = p.get("in_stock", 0)
        sku = p.get("sku", "")
        if sale_price and str(sale_price) not in ("0", "0.00", "None", ""):
            pricing = f"Cena: {price} RSD (na popustu: {sale_price} RSD)"
        else:
            pricing = f"Cena: {price} RSD"
        availability = "Dostupan" if int(in_stock or 0) > 0 else "Nije dostupan"
        line = f"• {name} — {pricing} — {availability}"
        if sku:
            line += f" (SKU: {sku})"
        lines.append(line)
    return "\n".join(lines)


def format_order_card(order: Dict, order_products: List[Dict]) -> str:
    oid = order.get("id", "?")
    status = _localize_status(order.get("status", ""))
    name = f"{order.get('first_name', '')} {order.get('last_name', '')}".strip()
    address_parts = [order.get("address", ""), order.get("city", ""), order.get("postcode", "")]
    address = ", ".join(p for p in address_parts if p)
    phone = order.get("phone", "")
    lines = [
        f"Narudžbina #{oid}",
        f"Status: {status}",
        f"Kupac: {name}",
        f"Adresa dostave: {address}",
        f"Telefon: {phone}",
        "Stavke:",
    ]
    for item in order_products:
        item_name = item.get("name") or item.get("short_name", "Nepoznat")
        lines.append(
            f"  • {item_name}"
            f" — Kol: {item.get('quantity')}"
            f" — Ukupno: {item.get('total')} RSD"
            f" (order_product_id={item['id']})"
        )
    if not order_products:
        lines.append("  (nema stavki)")
    return "\n".join(lines)


def format_order_context(order: Dict, order_products: List[Dict]) -> str:
    return format_order_card(order, order_products)


def format_product_context(products: List[Dict]) -> str:
    return format_product_card(products)


def build_system_prompt() -> str:
    return """Ti si Mango, asistent korisničke podrške za Clickshop online prodavnicu.

JEZIK: Odgovaraj isključivo na srpskom jeziku, latinicom. Nikada ne koristiš engleski ni bilo koji drugi jezik.

KLJUČNO PRAVILO: Koristi isključivo podatke koji su ti dostavljeni u SISTEMSKOM KONTEKSTU. Nikada ne izmišljaj, ne pretpostavljaj i ne dopunjavaj podatke o narudžbinama, proizvodima, cenama ili zalihama. Ako podatak nije u kontekstu, reci da ga nemaš.

--- PROIZVODI ---
- Odgovaraj samo na osnovu proizvoda dostavljenih u SISTEMSKOM KONTEKSTU.
- NIKADA ne izmišljaj ID-eve proizvoda ni cene. Koristi samo vrednosti koje se nalaze u dostavljenim podacima.
- Odgovaraj samo na pitanja vezana za prodavnicu i njene proizvode.
- Nikada ne navodi tačne količine na stanju. Kaži samo "dostupan" ili "nije dostupan".
- Ako je proizvod dostupan, predloži kupcu da naruči.
- Ako je proizvod nedostupan, jasno to saopšti i nemoj obećavati isporuku.
- Kada proizvod ima cenu na popustu, uvek navedi i redovnu i cenu na popustu.

--- NARUDŽBINE ---
- Narudžbine pretraži isključivo po ID broju koji kupac navede.
- Možeš da ažuriraš adresu dostave kada kupac navede nove podatke.
- Možeš da dodaš ili ukloniš proizvod iz narudžbine na zahtev kupca.
- Pre svake izmene narudžbine, obavezno zatraži potvrdu od kupca. Akciju izvršavaj tek po potvrdi.
- Narudžbine sa statusom 'poslata' ili 'isporučena' ne mogu se menjati.

--- AŽURIRANJE ADRESE ---
Pre generisanja JSON akcije za izmenu adrese, sva sledeća polja moraju biti poznata:
  - adresa (ulica i broj) — mora biti eksplicitno navedena od strane kupca
  - grad — mora biti eksplicitno naveden od strane kupca
  - poštanski broj — mora biti eksplicitno naveden; ako nije naveden, OBAVEZNO pitaj; NIKADA ne pretpostavljaj
  - telefon — ako kupac kaže da ostavi isti ili ga ne pomene, koristi postojeći broj iz podataka o narudžbini
  - ime i prezime — uvek preuzimaj iz postojeće narudžbine, osim ako kupac EKSPLICITNO traži promenu

Generiši JSON akciju tek kada su adresa, grad i poštanski broj eksplicitno potvrđeni od kupca.

--- IZVRŠAVANJE AKCIJA ---
Kada kupac potvrdi izmenu i svi potrebni podaci su poznati, odgovori SAMO jednim JSON blokom (bez teksta pre ili posle):

Za izmenu adrese:
{"action": "update_address", "order_id": 12345, "first_name": "Ime", "last_name": "Prezime", "address": "Ulica bb", "city": "Grad", "postcode": "00000", "phone": "0601234567"}

Za dodavanje proizvoda:
{"action": "add_product", "order_id": 12345, "product_id": 123, "quantity": 1, "total": 1435.0}

Za uklanjanje proizvoda:
{"action": "remove_product", "order_id": 12345, "order_product_id": 456}

Sistem će izvršiti akciju i obavestiti te o rezultatu. Tada potvrdi kupcu šta je urađeno jednom kratkom rečenicom."""
