"""The labels of a service request, in one place.

The categories and urgencies are the team's labels (docs/DATA_CARD.md): Farah's labeling guide
and test files spell them exactly like this. Her model outputs "pest control" (with a space);
the classifier service maps it to "pest_control" before anything is saved.

Who reads these lists: the database CHECK rules (app/db/models.py) and, later, the classifier
service and the session store.

Changing a value is more than an edit here: the database has CHECK rules with the old list, and
old rows may use the old value. It needs a hand-written migration, because Alembic's autogenerate
does not see CHECK changes. A label change is also a team contract change (docs/contracts.md).
"""

from typing import Literal, get_args

Category = Literal["plumbing", "electricity", "ac", "carpentry", "pest_control", "general"]
Urgency = Literal["low", "medium", "high"]
# The language of a request and of the bot's replies
Language = Literal["ar", "en"]
# Who wrote a message: the resident or the bot. "assistant", as in docs/contracts.md
MessageRole = Literal["user", "assistant"]

# The same values as tuples, for loops and CHECK rules (get_args reads them out of the Literal)
CATEGORIES: tuple[Category, ...] = get_args(Category)
URGENCIES: tuple[Urgency, ...] = get_args(Urgency)
LANGUAGES: tuple[Language, ...] = get_args(Language)
MESSAGE_ROLES: tuple[MessageRole, ...] = get_args(MessageRole)
