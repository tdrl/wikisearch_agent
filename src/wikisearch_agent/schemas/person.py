"""Schemas for structuring information about people and related names / named entities."""

from pydantic import BaseModel, Field


class PersonInfo(BaseModel):
      """Information about a single person-entity."""
      birth_name: str = Field(description='Full birth name')
      best_known_as: str = Field(description='Name by which this person is most widely known')
      alternate_names: list[str] = Field(description='List of other or alternate names, such as aliases, '
                                                     'pen names, noms de guerre, nicknames, etc.')
      best_known_for: str = Field(description='A one sentence description of why this person is '
                                              'noteworthy what they are known for')
      is_real: bool = Field(description='True iff this entity is real (as opposed to fictional, imaginary,'
                                        ' a character in a book or movie, etc.)')
      is_human: bool = Field(description='True iff this entity is a human (as opposed to, say, an animal, '
                                         'pet, alien, monster, imaginary being, etc.)')
      birth_year: int | None = Field(description='Year of birth (if known). Use a negative value for BCE '
                                                    'dates; positive for CE dates.')
      birth_month: int | None = Field(description='Month of birth (if known), starting with January = 1 '
                                                  'through December = 12.')
      birth_day: int | None = Field(description='Day of birth, within month, in the range [1, 31].')
      assigned_gender_at_birth: str = Field(description='Entitie''s gender, as assigned at birth. '
                                                        'Possible values: Male|Female|Nonbinary|Two spirit|Other|Unknown')
      gender_identity: str = Field(description='Entitie''s self-identified gender identity. May or '
                                               'may not be the same as their assigned gender at birth. Possible values: '
                                               'Male|Female|Nonbinary|Two spirit|Other|Unknown')
      continent_of_origin: str = Field(description='The continent or island on which they were born. '
                                                   'Possible values: Africa|Asia|Australia|Europe|South America|North '
                                                   'America|Antarctica|Island name')
      country_of_origin: str = Field(description='The country in which they were born (if any), as that '
                                                 'country was known at the time of their birth. Country '
                                                 'name, or null')
      locality_of_origin: str = Field(description='City or town or village name, or null')


class NameReference(BaseModel):
      """Records a single person's name, relationship, and a binding to a corresponding Wikipedia URL, if it has one."""
      name: str = Field(description="The person's name, as given in a single mention in the text "
                                    "of a Wikipedia article.")
      relationship: str | None = Field(description='Relationship of this named entity to the head entity of '
                                                   'the article. E.g., "mother", "brother", "employer", "opponent", etc. '
                                                   'Multi-word relationship descriptions are acceptable when necessary '
                                                   '(e.g., "brother and employee" or "third cousin, twice removed"), but '
                                                   'prefer a single word when possible. If the relationship is not explicitly '
                                                   'stated, or is unclear, don''t guess - just return null/None.')
      url: str | None = Field(description='The URL associated with the name mention, if one is '
                                          'present. null or None if the name is a bare mention, '
                                          'with no URL link.')


class ArticleNames(BaseModel):
      """Stores a list of all person names encountered in an article."""
      names: list[NameReference] = Field(description='List of all the names of people found in this article, with '
                                                     'relationship and associated URL, if any.')