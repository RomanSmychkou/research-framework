#Tasks.md
меня интересует: limitleveltree структура, order.uid и где он используется, почему levels возвращает только цены и где он используется

LimitLevel и LimitLevelTree не используют Order.uid
Order.uid прямо в Order не используются

Order-а добавляют сами себя (связный список)

LocalOrderBook:
• .update использует uid для смены значений без обновления parent
(Lazy compute)
• .remove использует uid для удаления с переназначением head/tail,
рекурсивной отчисткой limitlevel-ов и переназначением объёма уровня


Планы:
uid убрать, order сделать затычкой которая участвует лишь в дельте?
хотя лучше сделать как раз объект OrderBookDelta и не обзываться ордером