#!/usr/local/bin/python3.9

import ast
from pyclbr import Function
import simplify as si
import equality
import pprint2
from simplify import FALSE
from simplify import TRUE
import os

def _expand(kind, n: ast.AST):
    if isinstance(n, ast.BoolOp) and isinstance(n.op, kind):
        return [v for m in n.values for v in _expand(kind, m)]
    return [n]


def distr(expr1, expr2):
    if isinstance(expr1, ast.BoolOp) and isinstance(expr1.op, ast.And):
        return ast.BoolOp(ast.And(), values = [distr(expr1.values[0], expr2), distr(expr1.values[1], expr2)])
    if isinstance(expr2, ast.BoolOp) and isinstance(expr2.op, ast.And):
        return ast.BoolOp(ast.And(), values = [distr(expr1, expr2.values[0]), distr(expr1, expr2.values[1])])
    else:
        return ast.BoolOp(ast.Or(), values = [expr1, expr2])

def cnfp(expr):
    if isinstance(expr, (ast.Name, ast.Compare, ast.Call, ast.Attribute)):
        return expr
    if isinstance(expr, ast.UnaryOp):
        return expr
    if isinstance(expr, ast.BoolOp) and isinstance(expr.op, ast.And):
        return ast.BoolOp(ast.And(), values = [cnfp(expr.values[0]), cnfp(expr.values[1])])
    if isinstance(expr, ast.BoolOp) and isinstance(expr.op, ast.Or):
        return distr(cnfp(expr.values[0]), cnfp(expr.values[1]))
    
def nnf(expr):

    if isinstance(expr, ast.Call):
        return expr
    if isinstance(expr, (ast.Name, ast.Compare, ast.Attribute)):
        return expr
    if isinstance(expr, ast.UnaryOp):
        if isinstance(expr.operand, ast.UnaryOp):
            return expr.operand.operand
        
    if isinstance(expr, ast.BoolOp):
        return ast.BoolOp(op = expr.op, values = [nnf(x) for x in expr.values])

    if isinstance(expr, ast.UnaryOp):
        if isinstance(expr.operand, ast.BoolOp):
            if isinstance(expr.operand.op, ast.And):
                return nnf(ast.BoolOp(
                    op=ast.Or(),
                    values=[
                        ast.UnaryOp(op=ast.Not(), operand=expr.operand.values[0]),
                        ast.UnaryOp(op=ast.Not(), operand=expr.operand.values[1]),
                    ],
                ))
            elif isinstance(expr.operand.op, ast.Or):
                return nnf(ast.BoolOp(
                    op=ast.And(),
                    values=[
                        ast.UnaryOp(op=ast.Not(), operand=expr.operand.values[0]),
                        ast.UnaryOp(op=ast.Not(), operand=expr.operand.values[1]),
                    ],
                ))
    if isinstance(expr, ast.UnaryOp):
        return ast.UnaryOp(op=ast.Not(), operand=nnf(expr.operand))

def cnf_expr(expr):
    return cnfp(nnf(expr))
class Simplifier(ast.NodeTransformer):
    """
    - Collapse nested sequences of AND/OR BoolOp, e.g., (a and b) and (c and d).
    - Rewrite equality.Impl node `p => q === not p or q`
    - Collapse double negation, e.g.,  not (x not in [1,2,3])
    - Reduce simple BoolOp negations, e.g., not(a or b or c), not(not a) etc.
    - Convert tuple to list
    - Rewrite `x <= min(a, b) === x <= a and x <= b`
    """

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        
        

        def _split(kind, n):
            if len(n.values) > 2:
                start = ast.BoolOp(kind(), values = n.values[:2])
                for i in range(2, len(n.values)):
                    start = ast.BoolOp(kind(), values = [start, n.values[i]])
                return start
            else:
                return n
        self.generic_visit(node)
        if isinstance(node.op, ast.And):
            # values = _expand(ast.And, node)
            # return ast.BoolOp(op=ast.And(), values=values)
            return _split(ast.And, node)
        if isinstance(node.op, ast.Or):
            # values = _expand(ast.Or, node)
            # return ast.BoolOp(op=ast.Or(), values=values)
            return _split(ast.Or, node)
        if isinstance(node.op, equality.Impl):
            assert len(node.values) == 2
            return self.visit(
                ast.BoolOp(
                    op=ast.Or(),
                    values=[
                        ast.UnaryOp(op=ast.Not(), operand=node.values[0]),
                        node.values[1],
                    ],
                )
            )
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.op, ast.Not):
            if isinstance(node.operand, ast.UnaryOp) and isinstance(
                node.operand.op, ast.Not
            ):
                return node.operand.operand
            if isinstance(node.operand, ast.Compare):
                if len(node.operand.ops) == 1:
                    op = node.operand.ops[0]
                    comparators = node.operand.comparators
                    if isinstance(op, ast.NotEq):
                        return ast.Compare(
                            left=node.operand.left,
                            ops=[ast.Eq()],
                            comparators=comparators,
                        )
                    if isinstance(op, ast.Eq):
                        return ast.Compare(
                            left=node.operand.left,
                            ops=[ast.NotEq()],
                            comparators=comparators,
                        )
                    if isinstance(op, ast.NotIn):
                        return ast.Compare(
                            left=node.operand.left,
                            ops=[ast.In()],
                            comparators=comparators,
                        )
                    if isinstance(op, ast.In):
                        return ast.Compare(
                            left=node.operand.left,
                            ops=[ast.NotIn()],
                            comparators=comparators,
                        )
                    if isinstance(op, ast.IsNot):
                        return ast.Compare(
                            left=node.operand.left,
                            ops=[ast.Is()],
                            comparators=comparators,
                        )
                    if isinstance(op, ast.Is):
                        return ast.Compare(
                            left=node.operand.left,
                            ops=[ast.IsNot()],
                            comparators=comparators,
                        )
                    if isinstance(op, ast.Lt):
                        return ast.Compare(
                            left=node.operand.left,
                            ops=[ast.GtE()],
                            comparators=comparators,
                        )
                    if isinstance(op, ast.LtE):
                        return ast.Compare(
                            left=node.operand.left,
                            ops=[ast.Gt()],
                            comparators=comparators,
                        )
                    if isinstance(op, ast.Gt):
                        return ast.Compare(
                            left=node.operand.left,
                            ops=[ast.LtE()],
                            comparators=comparators,
                        )
                    if isinstance(op, ast.GtE):
                        return ast.Compare(
                            left=node.operand.left,
                            ops=[ast.Lt()],
                            comparators=comparators,
                        )
            if isinstance(node.operand, ast.BoolOp):
                values = [
                    self.visit(ast.UnaryOp(op=ast.Not(), operand=v))
                    for v in node.operand.values
                ]
                if isinstance(node.operand.op, ast.And):
                    return ast.BoolOp(op=ast.Or(), values=values)
                if isinstance(node.operand.op, ast.Or):
                    return ast.BoolOp(op=ast.And(), values=values)
                return node
            return node
        else:
            assert False
        return node

    def visit_Compare(self, node):
        self.generic_visit(node)
        comparisons = list(
            zip([node.left] + node.comparators, node.ops, node.comparators)
        )

        def min_max(node):
            assert len(node.ops) == 1 and len(node.comparators) == 1
            o = node.ops[0]
            c = node.comparators[0]
            if (
                isinstance(o, (ast.Lt, ast.LtE))
                and isinstance(c, ast.Call)
                and isinstance(c.func, ast.Name)
                and c.func.id == "min"
            ):
                values = [
                    ast.Compare(left=node.left, ops=[o], comparators=[c])
                    for c in c.args
                ]
                return ast.BoolOp(op=ast.And(), values=values)
            if (
                isinstance(o, (ast.Gt, ast.GtE))
                and isinstance(c, ast.Call)
                and isinstance(c.func, ast.Name)
                and c.func.id == "max"
            ):
                values = [
                    ast.Compare(left=node.left, ops=[o], comparators=[c])
                    for c in c.args
                ]
                return ast.BoolOp(op=ast.And(), values=values)
            return node

        values = [
            min_max(ast.Compare(left=l, ops=[o], comparators=[c]))
            for l, o, c in comparisons
        ]
        if len(values) == 1:
            return values[0]
        return ast.BoolOp(op=ast.And(), values=values)

    def visit_Tuple(self, node):
        return ast.List(elts=node.elts)

    def visit_Constant(self, node):
        return node

class FuncLocator(ast.NodeVisitor):
    def __init__(self, path, func, loc):
      self.path = path
      self.loc = loc
      self.func = func
      self.func_node = None
    def find(self):
      self.func_node = None
      with open(self.path, 'r',encoding="utf8") as src_file:
        src = src_file.read()
        tree = ast.parse(src, mode='exec')
        self.visit(tree)
      return self.func_node
    def visit_FunctionDef(self, node: ast.FunctionDef):
      if node.name == self.func and node.lineno <= self.loc and self.loc <= node.end_lineno:
        self.func_node = node
      super().generic_visit(node)

class AttrLocator(ast.NodeVisitor):
    def find(self, func_node, loc, attr):
        self.name = None
        self.loc = loc
        self.attr = attr
        self.visit(func_node)
        return self.name
    def visit_Attribute(self, node:ast.Attribute):
        if node.lineno == self.loc and node.attr == self.attr:
            if isinstance(node.value, ast.Name):
                self.name = node.value.id
        super().generic_visit(node)
# TOPPPPPP
class Analyzer(ast.NodeVisitor):
    def __init__(self, path):
      # function under analysis: fully qualified name: file:[Class|None]:name
      self.path = path
      # to track individual exceptions (either Raise or Call node)
      self.curr_ast_node = None
      self.curr_ast_node_key = None
      # wp stack, holds wp formulas        
      self.stack = []
      self.loc = None
      self.loc_found = False
      self.wp = None
      self.func_node = None
    def get_wp_for_loc(self, func, loc, attr):
      self.loc = loc
      self.loc_found = False
      self.wp = None
      self.stakc = []
      self.func_node = None
      func_node = FuncLocator(self.path, func, loc).find()
      if func_node:
        self.func_node = func_node
        self.visit(func_node)
      if self.wp:
        print(pprint2.pprint_top(self.wp))
        name = AttrLocator().find(func_node, loc, attr)
        return self.wp, name 
      return None, None
    def visit_FunctionDef(self, node):
      if node != self.func_node:
        return
      self._FunctionDef_Helper(node)


    def _FunctionDef_Helper(self, node):

      self.stack.append(TRUE);
      assert len(self.stack) == 1

      self.Body_Helper(node.body);
      
      wp = self.stack.pop();
      if self.loc_found:
        self.wp = wp


    def wp_from_two_arms(self, test, expr1, expr2):

        post = self.stack[-1]
        if isinstance(expr1, list):
            self.Body_Helper(expr1)
        else:
            self.visit(expr1)
        wp1 = self.stack.pop()

        self.stack.append(post)
        assert len(self.stack) == 1
        if isinstance(expr2, list):
            self.Body_Helper(expr2)
        else:
            self.visit(expr2)
        wp2 = self.stack.pop()

        if equality.compare(wp1,FALSE) and equality.compare(wp2,TRUE):
          # test => false and !test => true == !test 
          wp = si.negate(test);
        elif equality.compare(wp2,FALSE) and equality.compare(wp1,TRUE):
          # test => true and !test => false
          wp = test;
        elif equality.compare(wp1,wp2): 
          wp = wp1;
        elif equality.compare(wp2,FALSE):
          wp = si.cons_and(test,wp1); #ast.BoolOp(op=ast.And(),values=[node.test,wp1]) 
        elif equality.compare(wp1,FALSE):
          wp = si.cons_and(si.negate(test),wp2); 
        elif equality.compare(wp1,TRUE):
          # wp is !test => wp2 == test or wp2                                                                                              
          # wp = ast.BoolOp(op=ast.Or(),values=[node.test,wp2])
          wp = si.cons_impl(si.negate(test),wp2); # ast.BoolOp(op=Impl(),values=[negate(node.test),wp2])
        elif equality.compare(wp2,TRUE):         
          # test => wp1 == Not(test) or wp1                                                                          
          wp = si.cons_impl(test,wp1); # ast.BoolOp(op=Impl(),values=[node.test,wp1]);       
        else:       
          wp = si.factor_out(test,wp1,wp2);
        return wp
    def visit_If(self, node):
        wp = self.wp_from_two_arms(node.test, node.body, node.orelse)
        self.stack.append(wp)
        assert len(self.stack) == 1
        
        self.generic_visit(node)
    def visit_Raise(self, node):
      super(Analyzer, self).generic_visit(node);
      # self.stack.pop();
      # self.stack.append(TRUE);
      # assert len(self.stack) == 1

    # requires: body is a body, i.e., list of stmts                                                                                            
    def Body_Helper(self, body):
      for stmt in reversed(body):
        #print("\n!!!!!!! Here the stmt is:", ast.unparse(stmt),"\n")
        super(Analyzer, self).visit(stmt)

    def visit_Return(self, node):
        super(Analyzer, self).generic_visit(node);
        self.stack.pop();
        self.stack.append(TRUE);
        assert len(self.stack) == 1

    
    def visit_Expr(self, node):
        super(Analyzer, self).generic_visit(node);
    def visit_Name(self, node):
        self.RecordStruc(node)
        super(Analyzer, self).generic_visit(node)
    def visit_Attribute(self, node):
        self.RecordStruc(node)
        super(Analyzer, self).generic_visit(node)
    def visit_IfExp(self, node:ast.IfExp):
        # if node.lineno == self.loc:
        #     pass

        # wp = self.wp_from_two_arms(node.test, node.body, node.orelse)
        # self.stack.append(wp)

        self.RecordStruc(node)
        return super().generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp):
        # if node.lineno == self.loc:
        #     pass

        # # a or b == if a then a else b
        # if isinstance(node.op, ast.Or) and len(node.values) == 2:
        #     wp = self.wp_from_two_arms(node.values[0], node.values[0], node.values[1])
        #     self.stack.append(wp)

        
        self.RecordStruc(node)
        return super().generic_visit(node)
    def visit_AnnAssign(self, node):
        self.RecordStruc(node)
        super().generic_visit(node)

    def visit_AugAssign(self, node):
        self.RecordStruc(node)
        super().generic_visit(node)

    def visit_Assign(self, node):
        self.RecordStruc(node)
        return super().generic_visit(node)
    def visit_Global(self, node):
        self.RecordStruc(node)
        super().generic_visit(node)

    def visit_Nonlocal(self, node):
        self.RecordStruc(node)
        super().generic_visit(node)

    def visit_Pass(self, node):
        self.RecordStruc(node)
        super().generic_visit(node)
    def visit_Delete(self, node):
        self.RecordStruc(node)
        super().generic_visit(node)

    def visit_Raise(self, node):
        self.RecordStruc(node)
        super().generic_visit(node)

    def visit_Assert(self, node):
        self.RecordStruc(node)
        super().generic_visit(node)

    def visit_Call(self, node):
        self.RecordStruc(node)
        return super().generic_visit(node)

    def RecordStruc(self, node):
        if node.lineno == self.loc:
          self.stack.pop()
          self.stack.append(FALSE)
          assert len(self.stack) == 1
          self.loc_found = True



def split_line(line):
    loc, v, all_t, occur_t = line.strip().split('$$')
    l =  loc.strip().split('-')[-1]
    func = loc.strip().split('-')[-2]
    f = '-'.join(loc.strip().split('-')[:-2])
    return f, func, l, v, all_t, occur_t

# cond_target = 0
# no_cond_target = 0
# no_wp_cnt = 0
# with open('/home/user/thefuck-master/attr_load_events.txt') as f:
#     for line in f:
#         f, func, l, v, all_t, occur_t = split_line(line)
#         if os.path.exists(f):
#           analyzer = Analyzer(f)
#           wp = analyzer.get_wp_for_loc(func, int(l))
#           if wp:
#             if not isinstance(wp, ast.Constant):
#               swp = Simplifier().visit(wp)
#               cnf_wp = cnf_expr(swp)
#               values = _expand(ast.And, cnf_wp)
#               cnf_wp = ast.BoolOp(ast.And(), values)
#           else:
#             no_wp_cnt += 1
#             wp = analyzer.get_wp_for_loc(func, int(l))
# print(cond_target)

# print(no_cond_target)
# print(no_wp_cnt)