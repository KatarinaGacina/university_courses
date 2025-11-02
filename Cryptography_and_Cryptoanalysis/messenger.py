#!/usr/bin/env python3

from dataclasses import dataclass, field

from cryptography.hazmat.primitives.asymmetric import ec

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import dh
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives import (
    hashes,
)
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from cryptography.exceptions import InvalidTag

import pickle
import os
import sys

def gov_decrypt(gov_priv, message):
    """ TODO: Dekripcija poruke unutar kriptosustava javnog kljuca `Elgamal`
        gdje, umjesto kriptiranja jasnog teksta množenjem u Z_p, jasni tekst
        kriptiramo koristeci simetricnu sifru AES-GCM.

        Procitati poglavlje `The Elgamal Encryption Scheme` u udzbeniku
        `Understanding Cryptography` (Christof Paar , Jan Pelzl) te obratiti
        pozornost na `Elgamal Encryption Protocol`

        Dakle, funkcija treba:
        1. Izracunati `masking key` `k_M` koristeci privatni kljuc `gov_priv` i
           javni kljuc `gov_pub` koji se nalazi u zaglavlju `header`.
        2. Iz `k_M` derivirati kljuc `k` za AES-GCM koristeci odgovarajucu
           funkciju za derivaciju kljuca.
        3. Koristeci `k` i AES-GCM dekriptirati `gov_ct` iz zaglavlja da se
           dobije `sending (message) key` `mk`
        4. Koristeci `mk` i AES-GCM dekriptirati sifrat `ciphertext` orginalne
           poruke.
        5. Vratiti tako dobiveni jasni tekst.

        Naravno, lokalne varijable mozete proizvoljno imenovati.  Zaglavlje
        poruke `header` treba sadrzavati polja `gov_pub`, `gov_iv` i `gov_ct`.
        (mozete koristiti postojeci predlozak).

    """
    header, ciphertext = message

    k_M = gov_priv.exchange(ec.ECDH(), header.gov_pub)

    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=b"",
        info=b"key derivation function",
    )
    aes_key = hkdf.derive(k_M)

    aesgcm = AESGCM(aes_key)
    message_key = aesgcm.decrypt(header.gov_iv, header.gov_ct, None)

    aesgcm = AESGCM(message_key)
    return aesgcm.decrypt(header.iv, ciphertext, None).decode('utf-8')

# Možete se (ako želite) poslužiti sa sljedeće dvije strukture podataka
@dataclass
class Connection:
    dhs        : ec.EllipticCurvePrivateKey #sending ratchet key
    dhr        : ec.EllipticCurvePublicKey #receiving ratchet key
    rk         : bytes = None #root key
    cks        : bytes = None # chain keys sending
    ckr        : bytes = None
    pn         : int = 0 #number of messages in previous sending chain
    ns         : int = 0 #message number sending
    nr         : int = 0 #message number received
    mk_skipped : dict = field(default_factory=dict) #dict of skipped message keys indexed by ratchet public key and message number

@dataclass
class Header:
    rat_pub : bytes
    iv      : bytes
    gov_pub : bytes
    gov_iv  : bytes
    gov_ct  : bytes
    signature: bytes
    n       : int = 0
    pn      : int = 0

# Dopusteno je mijenjati sve osim sučelja.
class Messenger:
    """ Klasa koja implementira klijenta za čavrljanje
    """

    MAX_MSG_SKIP = 10
    
    initial_salt = os.urandom(32)

    #parameters = dh.generate_parameters(generator=2, key_size=2048)

    def __init__(self, username, ca_pub_key, gov_pub):
        """ Inicijalizacija klijenta

        Argumenti:
            username (str)      --- ime klijenta
            ca_pub_key (class)  --- javni ključ od CA (certificate authority)
            gov_pub (class) --- javni ključ od vlade

        Returns: None
        """
        self.username = username
        self.ca_pub_key = ca_pub_key
        self.gov_pub = gov_pub
        self.conns = {}

        self.verified_certificates = {}

        self.private_key = None
        self.public_key = None

    def generate_certificate(self):
        """ TODO: Metoda generira i vraća certifikacijski objekt.

        Metoda generira inicijalni par Diffie-Hellman ključeva. Serijalizirani
        javni ključ, zajedno s imenom klijenta, pohranjuje se u certifikacijski
        objekt kojeg metoda vraća. Certifikacijski objekt može biti proizvoljnog
        tipa (npr. dict ili tuple). Za serijalizaciju ključa možete koristiti
        metodu `public_bytes`; format (PEM ili DER) je proizvoljan.

        Certifikacijski objekt koji metoda vrati bit će potpisan od strane CA te
        će tako dobiveni certifikat biti proslijeđen drugim klijentima.

        Returns: <certificate object>
        """

        #self.private_key = Messenger.parameters.generate_private_key()
        #self.public_key = self.private_key.public_key()

        self.private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
        self.public_key = self.private_key.public_key()

        public_key_serialized = self.public_key.public_bytes(encoding=serialization.Encoding.PEM,
                                                             format=serialization.PublicFormat.SubjectPublicKeyInfo)
        #print(public_key_serialized.decode())

        return (public_key_serialized, self.username)

    def receive_certificate(self, cert_data, cert_sig):
        """ TODO: Metoda verificira certifikat od `CA` i sprema informacije o
                  klijentu.

        Argumenti:
        cert_data --- certifikacijski objekt
        cert_sig  --- digitalni potpis od `cert_data`

        Returns: None

        Metoda prima certifikat --- certifikacijski objekt koji sadrži inicijalni
        Diffie-Hellman javni ključ i ime klijenta s kojim želi komunicirati te njegov
        potpis. Certifikat se verificira pomoću javnog ključa CA (Certificate
        Authority), a ako verifikacija uspije, informacije o klijentu (ime i javni
        ključ) se pohranjuju. Javni ključ CA je spremljen tijekom inicijalizacije
        objekta.

        U slučaju da verifikacija ne prođe uspješno, potrebno je baciti iznimku.

        """

        try:
            self.ca_pub_key.verify(cert_sig, pickle.dumps(cert_data), ec.ECDSA(hashes.SHA256()))
            self.verified_certificates[cert_data[1]] = cert_data[0]

            username = cert_data[1]
            username_pk = serialization.load_pem_public_key(cert_data[0])

            connection = Connection(self.private_key, username_pk)

            shared_secret = connection.dhs.exchange(ec.ECDH(), connection.dhr)
            connection.rk = self.__hkdf_step(Messenger.initial_salt, 32, shared_secret) 

            self.conns[username] = connection 

        except Exception:
            raise Exception("Exception!")

        return

    def __hkdf_step(self, salt, length, shared_secret):
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=length,
            salt=salt,
            info=b"key derivation function",
        )
        return hkdf.derive(shared_secret)

    def send_message(self, username, message):
        """ TODO: Metoda šalje kriptiranu poruku `message` i odgovarajuće
                  zaglavlje korisniku `username`.

        Argumenti:
        message  --- poruka koju ćemo poslati
        username --- korisnik kojem šaljemo poruku

        returns: (header, ciphertext).

        Zaglavlje poruke treba sadržavati podatke potrebne
        1) klijentu da derivira nove ključeve i dekriptira poruku;
        2) Velikom Bratu da dekriptira `sending` ključ i dode do sadržaja poruke.

        Pretpostavite da već posjedujete certifikacijski objekt klijenta (dobiven
        pomoću metode `receive_certificate`) i da klijent posjeduje vaš. Ako
        prethodno niste komunicirali, uspostavite sesiju generiranjem ključeva po-
        trebnih za `Double Ratchet` prema specifikaciji. Inicijalni korijenski ključ
        (`root key` za `Diffie-Hellman ratchet`) izračunajte pomoću ključa
        dobivenog u certifikatu i vašeg inicijalnog privatnog ključa.

        Svaka poruka se sastoji od sadržaja i zaglavlja. Svaki put kada šaljete
        poruku napravite korak u lancu `symmetric-key ratchet` i lancu
        `Diffie-Hellman ratchet` ako je potrebno prema specifikaciji (ovo drugo
        možete napraviti i prilikom primanja poruke); `Diffie-Helman ratchet`
        javni ključ oglasite putem zaglavlja. S novim ključem za poruke
        (`message key`) kriptirajte i autentificirajte sadržaj poruke koristeći
        simetrični kriptosustav AES-GCM; inicijalizacijski vektor proslijedite
        putem zaglavlja. Dodatno, autentificirajte odgovarajuća polja iz
        zaglavlja, prema specifikaciji. (autentifikacija!)

        Sve poruke se trebaju moći dekriptirati uz pomoć privatnog kljuca od
        Velikog brata; pripadni javni ključ dobiti ćete prilikom inicijalizacije
        kli- jenta. U tu svrhu koristite protokol enkripcije `ElGamal` tako da,
        umjesto množenja, `sending key` (tj. `message key`) kriptirate pomoću
        AES-GCM uz pomoć javnog ključa od Velikog Brata. Prema tome, neka
        zaglavlje do- datno sadržava polja `gov_pub` (`ephemeral key`) i
        `gov_ct` (`ciphertext`) koja predstavljaju izlaz `(k_E , y)`
        kriptosustava javnog kljuca `Elgamal` te `gov_iv` kao pripadni
        inicijalizacijski vektor.

        U ovu svrhu proučite `Elgamal Encryption Protocol` u udžbeniku
        `Understanding Cryptography` (glavna literatura). Takoder, pročitajte
        dokumentaciju funkcije `gov_decrypt`.

        Za zaglavlje možete koristiti već dostupnu strukturu `Header` koja sadrži
        sva potrebna polja.

        Metoda treba vratiti zaglavlje i kriptirani sadrzaj poruke kao `tuple`:
        (header, ciphertext).

        """

        if username not in self.conns:
            raise Exception("Exception!")
            
        connection = self.conns[username]

        if (connection.ns == 0): #connection.nr != 0 and
            connection.dhs = ec.generate_private_key(ec.SECP256R1(), default_backend())

            shared_secret = connection.dhs.exchange(ec.ECDH(), connection.dhr)
            kljucevi = self.__hkdf_step(connection.rk, 32*2, shared_secret) 
            connection.rk, connection.cks = kljucevi[:32], kljucevi[32:]

        connection.cks = message_key = self.__hkdf_step(connection.rk, 32, connection.cks)

        aesgcm = AESGCM(message_key)

        #for goverment
        iv = os.urandom(12)
        ciphertext = aesgcm.encrypt(iv, message.encode('utf-8'), None)

        curve = self.gov_pub.curve
        random_key = ec.generate_private_key(curve)

        k_E = random_key.public_key() #r * G
        k_M = random_key.exchange(ec.ECDH(), self.gov_pub)

        aes_key = self.__hkdf_step(b"", 32, k_M)
        aesgcm_gov = AESGCM(aes_key)

        gov_iv = os.urandom(12)
        gov_ciphertext = aesgcm_gov.encrypt(gov_iv, message_key, None)


        rat_pub_ser = connection.dhs.public_key().public_bytes(encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo)
        signature = self.private_key.sign(pickle.dumps((connection.ns, connection.pn, rat_pub_ser)), ec.ECDSA(hashes.SHA256())) #gcm

        header = Header(connection.dhs.public_key(), iv, k_E, gov_iv, gov_ciphertext, signature, connection.ns, connection.pn)

        connection.ns += 1

        """print(username)
        print(connection.rk)
        print(connection.cks)
        print(connection.ckr)
        print()"""

        return (header, ciphertext)
    
    def __skiped_messages(self, connection, key, n):
        if (connection.nr + Messenger.MAX_MSG_SKIP < n):
            raise Exception()

        while connection.nr < n:
            message_key = connection.ckr = self.__hkdf_step(connection.rk, 32, connection.ckr)
            
            if (len(connection.mk_skipped) >= Messenger.MAX_MSG_SKIP):
                raise Exception("Exception!")

            connection.mk_skipped[(key, connection.nr)] = message_key

            connection.nr += 1

    def receive_message(self, username, message):
        """ TODO: Primanje poruke od korisnika

        Argumenti:
        message  -- poruka koju smo primili
        username -- korisnik koji je poslao poruku

        returns: plaintext

        Metoda prima kriptiranu poruku od korisnika s imenom `username`.
        Pretpostavite da već posjedujete certifikacijski objekt od korisnika
        (dobiven pomoću `receive_certificate`) i da je korisnik izračunao
        inicijalni `root` ključ uz pomoć javnog Diffie-Hellman ključa iz vašeg
        certifikata.  Ako već prije niste komunicirali, uspostavite sesiju tako
        da generirate nužne `double ratchet` ključeve prema specifikaciji.

        Svaki put kada primite poruku napravite `ratchet` korak u `receiving`
        lanacu (i `root` lanacu ako je potrebno prema specifikaciji) koristeći
        informacije dostupne u zaglavlju i dekriptirajte poruku uz pomoć novog
        `receiving` ključa. Ako detektirate da je integritet poruke narušen,
        zaustavite izvršavanje programa i generirajte iznimku.

        Metoda treba vratiti dekriptiranu poruku.

        """

        if username not in self.conns:
            raise Exception("Exception!")
        
        header, ciphertext = message

        serialized_rat_pub = header.rat_pub.public_bytes(encoding=serialization.Encoding.PEM,
                                                format=serialization.PublicFormat.SubjectPublicKeyInfo)

        signature_key = serialization.load_pem_public_key(self.verified_certificates[username])

        try:
            signature_key.verify(header.signature, pickle.dumps((header.n, header.pn, serialized_rat_pub)), ec.ECDSA(hashes.SHA256()))
        except Exception:
            raise Exception("Exception!")
            
        connection = self.conns[username]

        if (serialized_rat_pub, header.n) in connection.mk_skipped:
            message_key = connection.mk_skipped[(serialized_rat_pub, header.n)]

            del connection.mk_skipped[(serialized_rat_pub, header.n)]

            aesgcm = AESGCM(message_key)
            return aesgcm.decrypt(header.iv, ciphertext, None).decode('utf-8')

        if (connection.dhr != header.rat_pub):
            self.__skiped_messages(connection, serialized_rat_pub, header.pn)

            connection.pn = connection.ns
            connection.ns = 0
            connection.nr = 0
            connection.dhr = header.rat_pub

            shared_secret = connection.dhs.exchange(ec.ECDH(), connection.dhr)
            kljucevi = self.__hkdf_step(connection.rk, 32*2, shared_secret) 
            connection.rk, connection.ckr = kljucevi[:32], kljucevi[32:]

        self.__skiped_messages(connection, serialized_rat_pub, header.n)

        message_key = connection.ckr = self.__hkdf_step(connection.rk, 32, connection.ckr)

        connection.nr += 1   

        """print(username)
        print(connection.rk)
        print(connection.cks)
        print(connection.ckr)
        print()"""     

        aesgcm = AESGCM(message_key)
        try:
            result = aesgcm.decrypt(header.iv, ciphertext, None).decode('utf-8')
        except InvalidTag:
            raise Exception("Exception!")
        
        return result

def main():
    pass

if __name__ == "__main__":
    main()
