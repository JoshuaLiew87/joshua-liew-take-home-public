-- Promote the Markdown Abstract section into Pandoc front matter so the
-- generated PDF places it before the table of contents.
-- PDF cannot animate GIFs, so the live-demo GIF is replaced by its
-- PDF-compatible first-frame PNG during PDF generation.
-- Wide report tables receive PDF-only sizing here so the Markdown remains
-- portable on GitHub and other renderers that display raw LaTeX commands.
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
  local current_section = ""

  local table_style = {
    ["10. assumptions and limitations"] = {
      before = "\\begingroup\\scriptsize\\sloppy\\renewcommand{\\arraystretch}{0.78}\\setlength{\\tabcolsep}{3pt}\\exhyphenpenalty=10000\\hyphenpenalty=10000",
      after = "\\endgroup",
    },
    ["12. assignment traceability"] = {
      before = "\\begingroup\\small\\sloppy\\renewcommand{\\arraystretch}{0.86}\\setlength{\\tabcolsep}{3pt}",
      after = "\\endgroup",
    },
  }

  local function insert_body_block(block)
    if block.t == "Table" and table_style[current_section] ~= nil then
      local style = table_style[current_section]
      if current_section == "10. assumptions and limitations" then
        -- Pandoc derives the first column from the short Markdown header
        -- "ID", which leaves too little room for identifiers such as A-001.
        -- Give the PDF table a stable first-column width while leaving the
        -- source Markdown portable for GitHub and other renderers.
        block.colspecs = {
          {pandoc.AlignDefault, 0.07},
          {pandoc.AlignDefault, 0.32},
          {pandoc.AlignDefault, 0.08},
          {pandoc.AlignDefault, 0.22},
          {pandoc.AlignDefault, 0.18},
          {pandoc.AlignDefault, 0.13},
        }
      end
      body:insert(pandoc.RawBlock("latex", style.before))
      body:insert(block)
      body:insert(pandoc.RawBlock("latex", style.after))
    else
      body:insert(block)
    end
  end

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
      current_section = pandoc.utils.stringify(block.content):lower()
      insert_body_block(block)
    elseif in_abstract then
      abstract:insert(block)
    else
      if block.t == "Header" and block.level == 2 then
        current_section = pandoc.utils.stringify(block.content):lower()
      end
      insert_body_block(block)
    end
  end

  if found_abstract then
    doc.meta.abstract = pandoc.MetaBlocks(abstract)
    doc.blocks = body
  end

  return doc
end
