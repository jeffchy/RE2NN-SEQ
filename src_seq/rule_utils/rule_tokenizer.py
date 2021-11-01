from pyparsing import Literal, Word, alphas, Optional, alphanums, OneOrMore, Forward, Group, ZeroOrMore, printables, Literal, Empty, oneOf, nums, ParserElement, Combine
from pydash import flatten_deep
from pyparsing import pyparsing_unicode as PU



def ruleParser(ruleString, lang='en'):
    WW = Word(PU.printables, excludeChars='$%&*()|?+<>{}:')
    ParserElement.enablePackrat()
    # WildCards = oneOf("$ % &") + "<:>" + Word(alphas)
    # LeafWord = WildCards | Word(alphas + "'`") |  Word(alphas) + "<:>" + Word(alphas)
    # $ means words, % means numbers, & means punctuations
    WildCards = Combine(oneOf("$ % &") + "<:>" + Word(alphas + "'`_-."))
    # NormalWords = Combine(Word(alphanums + "'`_-.") + "<:>" + Word(alphanums + "'`_-."))
    NormalWords_ZH = Combine(WW + "<:>" + Word(alphanums + "'`_-."))
    NormalWords_EN = Combine(Word(alphanums + "'`_-.") + "<:>" + Word(alphanums + "'`_-."))

    if lang == 'en':
        LeafWord = WildCards | NormalWords_EN
    elif lang == 'zh':
        LeafWord = WildCards | NormalWords_ZH
    else:
        raise NotImplementedError()

    # aaa+ aaa* aaa? aaa{0,3} aaa{2}
    RangedQuantifiers = Literal("{") + Word(nums) + Optional(
        Literal(",") + Word(nums)) + Literal("}")
    Quantifiers = oneOf("* + ?") | RangedQuantifiers
    QuantifiedLeafWord = LeafWord + Quantifiers
    # a sequence
    ConcatenatedSequence = OneOrMore(QuantifiedLeafWord | LeafWord)
    # syntax root
    Rule = Forward()
    # ( xxx )
    GroupStatement = Forward()
    QuantifiedGroup = GroupStatement + Quantifiers
    CaptureGroupStatement = Forward()
    # xxx | yyy
    orAbleStatement = QuantifiedGroup | GroupStatement | ConcatenatedSequence
    OrStatement = Group(orAbleStatement +
                        OneOrMore(Literal("|") + Group(orAbleStatement)))

    GroupStatement << Group(Literal("(") + Rule + Literal(")"))
    CaptureGroupStatement << Group(Literal("(") + Rule + Literal(")"))
    Rule << OneOrMore(OrStatement | orAbleStatement | CaptureGroupStatement)

    return flatten_deep(Rule.parseString(ruleString).asList())

if __name__ == "__main__":
    ruleStringSmall = '(( $<:>O * %<:>Rating ( star<:>Rating | starts<:>Rating ) $<:>O * ) | ( $<:>O * %<:>Rating ( star<:>Rating | starts<:>Rating ) $<:>O * ))'
    rule = "( $<:>O * %<:>Rating (star<:>Rating | starts<:>Rating ) $<:>O*) | ( $<:>O * ( good<:>Rating for<:>Amenity group<:>Amenity )|(inside<:>Amenity dining<:>Amenity )$<:>O * ) | ( $<:>O * ( salad<:>Dish | soft<:>Dish serve<:>Dish  ice<:>Dish cream<:>Dish | steak<:>Dish | salmon<:>Dish | chili<:>Dish | fresh<:>Dish fruits<:>Dish ) $<:>O * ) | ( $<:>O * ( within<:>Location a<:>Location mile<:>Location | in<:>Location my<:>Location town<:>Location | nearest<:>Location | supermarket<:>Location ) $<:>O * ) | ( $<:>O * ( davidos<:>Restaurant_Name italian<:>Restaurant_Name palace<:>Restaurant_Name | johns<:>Restaurant_Name pizza<:>Restaurant_Name cafe<:>Restaurant_Name | cracker<:>Restaurant_Name barrel<:>Restaurant_Name | passims<:>Restaurant_Name kitchen<:>Restaurant_Name ) $<:>O * ) | ( $<:>O * ( open<:>Hours till<:>Hours %<:>Hours ( a<:>Hours m<:>Hours | am<:>Hours ) | after<:>Hours hour<:>Hours dining<:>Hours | dinner<:>Hours per<:>Price person<:>Price ) $<:>O * ) | ( $<:>O * midpriced<:>Price bottle<:>Rating of<:>O good<:>Rating wine<:>Cuisine $<:>O * ) | ( $<:>O * smoothie<:>Cuisine $<:>O * )"
    parseResult = ruleParser(rule)
    print(parseResult)
