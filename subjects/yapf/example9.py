from typing import Optional
class FormatToken(object):

  def __init__(self, node, name):
    """Constructor.

    Arguments:
      node: (pytree.Leaf) The node that's being wrapped.
      name: (string) The name of the node.
    """
    self.node = node
    self.name = name
    self.type = node.type
    self.column = node.column
    self.lineno = node.lineno
    self.value = node.value

class FormatDecisionState(object):
  """The current state when indenting a logical line.

  The FormatDecisionState object is meant to be copied instead of referenced.

  Attributes:
    first_indent: The indent of the first token.
    column: The number of used columns in the current line.
    line: The logical line we're currently processing.
    next_token: The next token to be formatted.
    paren_level: The level of nesting inside (), [], and {}.
    lowest_level_on_line: The lowest paren_level on the current line.
    stack: A stack (of _ParenState) keeping track of properties applying to
      parenthesis levels.
    comp_stack: A stack (of ComprehensionState) keeping track of properties
      applying to comprehensions.
    param_list_stack: A stack (of ParameterListState) keeping track of
      properties applying to function parameter lists.
    ignore_stack_for_comparison: Ignore the stack of _ParenState for state
      comparison.
    column_limit: The column limit specified by the style.
  """
  next_token: Optional[FormatToken]
  def __init__(self, line, first_indent):
    """Initializer.

    Initializes to the state after placing the first token from 'line' at
    'first_indent'.

    Arguments:
      line: (LogicalLine) The logical line we're currently processing.
      first_indent: (int) The indent of the first token.
    """
    self.next_token = line.first
    self.column = first_indent
    self.line = line
    self.paren_level = 0
    self.lowest_level_on_line = 0
    self.ignore_stack_for_comparison = False
    self.stack = [_ParenState(first_indent, first_indent)]
    self.comp_stack = []
    self.param_list_stack = []
    self.first_indent = first_indent
    self.column_limit = style.Get('COLUMN_LIMIT')

def testSimpleFunctionDefWithNoSplitting(self):
    code = textwrap.dedent(r"""
      def f(a, b):
        pass
      """)
    llines = yapf_test_helper.ParseAndUnwrap(code)
    lline = logical_line.LogicalLine(0, _FilterLine(llines[0]))
    lline.CalculateFormattingInformation()

    # Add: 'f'
    state:FormatDecisionState = format_decision_state.FormatDecisionState(lline, 0)
    state.MoveStateToNextToken()
    self.assertEqual('f', state.next_token.value)
    self.assertFalse(state.CanSplit(False))

    # Add: '('
    state.AddTokenToState(False, True)
    self.assertEqual('(', state.next_token.value)
    self.assertFalse(state.CanSplit(False))
    self.assertFalse(state.MustSplit())

    # Add: 'a'
    state.AddTokenToState(False, True)
    self.assertEqual('a', state.next_token.value)
    self.assertTrue(state.CanSplit(False))
    self.assertFalse(state.MustSplit())

    # Add: ','
    state.AddTokenToState(False, True)
    self.assertEqual(',', state.next_token.value)
    self.assertFalse(state.CanSplit(False))
    self.assertFalse(state.MustSplit())

    # Add: 'b'
    state.AddTokenToState(False, True)
    self.assertEqual('b', state.next_token.value)
    self.assertTrue(state.CanSplit(False))
    self.assertFalse(state.MustSplit())

    # Add: ')'
    state.AddTokenToState(False, True)
    self.assertEqual(')', state.next_token.value)
    self.assertTrue(state.CanSplit(False))
    self.assertFalse(state.MustSplit())

    # Add: ':'
    state.AddTokenToState(False, True)
    self.assertEqual(':', state.next_token.value)
    self.assertFalse(state.CanSplit(False))
    self.assertFalse(state.MustSplit())

    clone = state.Clone()
    self.assertEqual(repr(state), repr(clone))