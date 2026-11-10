FIELDS = ["compound", "building", "unit_number", "name", "phone", "issue"]
UNIT_FIELDS = ["compound", "building", "unit_number"]

LABELS = {
    "en": {"compound": "compound", "building": "building", "unit_number": "unit number",
           "name": "your name", "phone": "your phone number", "issue": "a description of the issue"},
    "ar": {"compound": "اسم المجمع", "building": "رقم المبنى", "unit_number": "رقم الوحدة",
           "name": "اسمك", "phone": "رقم التليفون", "issue": "وصف المشكلة"},
}

TEXT = {
    "en": {
        "saved_unit": "Is this for {compound}, Building {building}, Unit {unit_number}? (yes / no)",
        "confirm": ("Please confirm your request:\n• Unit: {compound}, Building {building}, Unit {unit_number}\n"
                    "• Name: {name}\n• Phone: {phone}\n• Issue: {issue}\n\nIs this correct? (yes / no)"),
        "done": ("✅ Your request was submitted.\nTicket: {ticket}\nCategory: {category}\n"
                 "Urgency: {urgency}\n{review}\nCheck the status any time with /status {ticket}"),
        "review_note": "A team member will review the classification.",
        "what_change": "What would you like to change?",
        "missing_fallback": "Thanks! I still need: {items}.",
        "voice_unclear": "I couldn't understand the voice note. Please try again or type your message.",
        "status": "Ticket {ticket}: {status}\nCategory: {category} | Urgency: {urgency}\nCreated: {created}",
        "status_none": "I couldn't find that ticket for your account.",
        "status_usage": "Usage: /status SR-000001",
    },
    "ar": {
        "saved_unit": "هل الطلب ده للوحدة دي؟ {compound} - مبنى {building} - وحدة {unit_number}\n(نعم / لا)",
        "confirm": ("من فضلك أكد الطلب:\n• الوحدة: {compound} - مبنى {building} - وحدة {unit_number}\n"
                    "• الاسم: {name}\n• التليفون: {phone}\n• المشكلة: {issue}\n\nهل البيانات صحيحة؟ (نعم / لا)"),
        "done": ("✅ تم تسجيل طلبك.\nرقم الطلب: {ticket}\nالتصنيف: {category}\n"
                 "الأولوية: {urgency}\n{review}\nتقدر تتابع الحالة بـ /status {ticket}"),
        "review_note": "هيتم مراجعة التصنيف من أحد أعضاء الفريق.",
        "what_change": "إيه اللي تحب تعدله؟",
        "missing_fallback": "شكراً! لسه محتاج: {items}.",
        "voice_unclear": "مقدرتش أفهم الرسالة الصوتية. جرب تاني أو اكتب رسالتك.",
        "status": "الطلب {ticket}: {status}\nالتصنيف: {category} | الأولوية: {urgency}\nتاريخ الإنشاء: {created}",
        "status_none": "مش لاقي الطلب ده على حسابك.",
        "status_usage": "الاستخدام: /status SR-000001",
    },
}

RESET_DONE = "Session cleared. / تم مسح المحادثة الحالية."
FORGET_DONE = "Your saved profile was deleted. / تم حذف بياناتك المحفوظة."


def t(lang: str, key: str, **kw) -> str:
    return TEXT.get(lang, TEXT["en"])[key].format(**kw)