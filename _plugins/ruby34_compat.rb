# Compat shim for running the old github-pages / Jekyll 3.9 stack on Ruby 3.2+
# (Ruby removed the taint API in 3.2; liquid 4.0.3 still calls String#untaint).
# Local preview only — GitHub Pages builds server-side and ignores _plugins.
class Object
  def untaint; self; end unless method_defined?(:untaint)
  def taint; self; end unless method_defined?(:taint)
  def tainted?; false; end unless method_defined?(:tainted?)
end
