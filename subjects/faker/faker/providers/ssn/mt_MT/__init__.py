from .. import Provider as BaseProvider


class Provider(BaseProvider):
    """
    A Faker provider for the Maltese VAT IDs
    """

    vat_id_formats = ("MT########",)

    def vat_id(self) -> str:
        """
        http://ec.europa.eu/taxation_customs/vies/faq.html#item_11
        :return: A random Maltese VAT ID
        """

        return self.bothify(self.random_element(self.vat_id_formats))
