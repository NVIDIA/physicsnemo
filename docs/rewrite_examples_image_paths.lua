-- rewrite_image_paths.lua
function Image(el)
    -- Example: change '../../docs/img/foo.png' → '../../img/foo.png'
    el.src = string.gsub(el.src, "/docs/img/", "/img/")
    return el
  end
  