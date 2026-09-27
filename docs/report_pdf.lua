-- Promote the Markdown Abstract section into Pandoc front matter so the
-- generated PDF places it before the table of contents.
-- PDF cannot animate GIFs, so the live-demo GIF is replaced by its
-- PDF-compatible first-frame PNG during PDF generation.
function Image(image)
  if image.src:match("live_demo%.gif$") then
    image.src = image.src:gsub("live_demo%.gif$", "live_demo.png")
  end
  return image
end

function Pandoc(doc)
  local body = pandoc.List()
  local abstract = pandoc.List()
  local in_abstract = false
  local found_abstract = false
  local found_title = false

  for _, block in ipairs(doc.blocks) do
    if block.t == "Header" and block.level == 1 and not found_title then
      doc.meta.title = pandoc.MetaInlines(block.content)
      found_title = true
    elseif block.t == "Header"
        and block.level == 2
        and pandoc.utils.stringify(block.content):lower() == "abstract" then
      in_abstract = true
      found_abstract = true
    elseif in_abstract and block.t == "Header" and block.level <= 2 then
      in_abstract = false
      body:insert(block)
    elseif in_abstract then
      abstract:insert(block)
    else
      body:insert(block)
    end
  end

  if found_abstract then
    doc.meta.abstract = pandoc.MetaBlocks(abstract)
    doc.blocks = body
  end

  return doc
end
