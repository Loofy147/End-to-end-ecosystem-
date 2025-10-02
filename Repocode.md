







import React, { useState, useMemo, useEffect } from 'react'; import { Search, Star, Code, GitBranch, Zap, Eye, Filter, ExternalLink, Heart, Moon, Sun, ChevronLeft, ChevronRight, Loader, RefreshCw, X } from 'lucide-react';

const DevResourcesExplorer = () => { const [searchTerm, setSearchTerm] = useState(''); const [selectedCategory, setSelectedCategory] = useState('all'); const [favorites, setFavorites] = useState([]); const [darkMode, setDarkMode] = useState(false); const [sortBy, setSortBy] = useState('name'); const [currentPage, setCurrentPage] = useState(1); const [selectedResource, setSelectedResource] = useState(null); const [isGeneratingPractices, setIsGeneratingPractices] = useState(false); const [generatedPractices, setGeneratedPractices] = useState({}); const itemsPerPage = 6;

const resources = [ { id: 1, name: 'GitHub', category: 'repositories', description: 'أكبر مستودع للأكواد مفتوحة المصدر في العالم', url: 'https://github.com', rating: 5, features: ['مشاريع مفتوحة المصدر', 'التحكم في النسخ', 'التعاون الجماعي', 'CI/CD'], tags: ['git', 'open-source', 'collaboration'], popularity: 'عالية جداً', fullDescription: 'GitHub هي أكبر منصة لاستضافة وتطوير البرمجيات في العالم، توفر أدوات شاملة للتحكم في النسخ والتعاون والتطوير المستمر.', practiceArea: 'version-control' }, { id: 2, name: 'TheAlgorithms', category: 'algorithms', description: 'مكتبة ضخمة من الخوارزميات بلغات برمجة متعددة', url: 'https://github.com/TheAlgorithms', rating: 5, features: ['Python', 'Java', 'C++', 'JavaScript', 'Go'], tags: ['algorithms', 'data-structures', 'programming'], popularity: 'عالية', fullDescription: 'مجموعة شاملة من تنفيذ الخوارزميات المختلفة بلغات برمجة متعددة مع شروحات واضحة وأمثلة عملية.', practiceArea: 'algorithms' }, { id: 3, name: 'LeetCode Solutions', category: 'algorithms', description: 'حلول مسائل البرمجة والخوارزميات الشائعة', url: 'https://leetcode.com', rating: 5, features: ['مسائل متدرجة', 'شروحات مفصلة', 'مقابلات العمل', 'منافسات'], tags: ['interview-prep', 'algorithms', 'problem-solving'], popularity: 'عالية جداً', fullDescription: 'منصة رائدة لحل مسائل البرمجة والتحضير لمقابلات العمل في أكبر الشركات التقنية.', practiceArea: 'problem-solving' }, { id: 4, name: 'CodePen', category: 'sharing', description: 'محرر تفاعلي لـ HTML, CSS, JavaScript', url: 'https://codepen.io', rating: 4, features: ['محرر مباشر', 'مشاركة سهلة', 'مجتمع نشط', 'أمثلة متنوعة'], tags: ['frontend', 'css', 'javascript'], popularity: 'عالية', fullDescription: 'بيئة تطوير أمامية تفاعلية تسمح بكتابة وتجريب ومشاركة أكواد الويب بشكل مباشر.', practiceArea: 'frontend-development' }, { id: 5, name: 'Replit', category: 'cloud-ide', description: 'بيئة تطوير متكاملة في المتصفح', url: 'https://replit.com', rating: 5, features: ['50+ لغة برمجة', 'تطوير تشاركي', 'استضافة مجانية', 'قواعد بيانات'], tags: ['ide', 'collaboration', 'hosting'], popularity: 'عالية جداً', fullDescription: 'بيئة تطوير سحابية شاملة تدعم أكثر من 50 لغة برمجة مع إمكانيات التعاون والاستضافة المجانية.', practiceArea: 'cloud-development' }, { id: 6, name: 'Codeforces', category: 'challenges', description: 'منصة مسابقات البرمجة التنافسية', url: 'https://codeforces.com', rating: 5, features: ['مسابقات دورية', 'تصنيف عالمي', 'مسائل متدرجة', 'مجتمع نشط'], tags: ['competitive-programming', 'contests', 'algorithms'], popularity: 'عالية', fullDescription: 'أحد أشهر مواقع البرمجة التنافسية في العالم، يقدم مسابقات دورية ونظام تصنيف متقدم.', practiceArea: 'competitive-programming' } ];

const categories = { all: 'جميع الفئات', repositories: 'مستودعات الأكواد', algorithms: 'الخوارزميات', sharing: 'مشاركة الأكواد', 'cloud-ide': 'بيئات التطوير', visualization: 'أدوات التصور', challenges: 'التحديات', learning: 'التعلم', 'data-science': 'علم البيانات' };

useEffect(() => { const stored = localStorage.getItem('favorites'); if (stored) setFavorites(JSON.parse(stored)); }, []);

useEffect(() => { localStorage.setItem('favorites', JSON.stringify(favorites)); }, [favorites]);

const toggleFavorite = (id) => setFavorites(prev => prev.includes(id) ? prev.filter(f => f !== id) : [...prev, id]); const toggleDarkMode = () => setDarkMode(d => !d);

const generatePracticalPractices = async (resource) => { setIsGeneratingPractices(true); // API call logic here setIsGeneratingPractices(false); };

const filtered = useMemo(() => { return resources .filter(r => (r.name + r.description + r.tags.join(' ')).toLowerCase().includes(searchTerm.toLowerCase())) .filter(r => selectedCategory === 'all' || r.category === selectedCategory) .sort((a, b) => sortBy === 'rating' ? b.rating - a.rating : sortBy === 'popularity' ? ( {'عالية جداً':3,'عالية':2,'متوسطة':1}[b.popularity] - {'عالية جداً':3,'عالية':2,'متوسطة':1}[a.popularity] ) : a.name.localeCompare(b.name)); }, [searchTerm, selectedCategory, sortBy]);

const totalPages = Math.ceil(filtered.length / itemsPerPage); const pageItems = filtered.slice((currentPage-1)itemsPerPage, currentPageitemsPerPage);

const theme = darkMode ? 'min-h-screen bg-gray-900 text-white' : 'min-h-screen bg-slate-50 text-gray-800'; const card = darkMode ? 'bg-gray-800 border border-gray-700' : 'bg-white border border-gray-100';

return ( <div className={theme} dir="rtl"> <div className="max-w-7xl mx-auto p-6"> {/* Header */} <div className="text-center mb-8"> <h1 className="text-4xl font-bold inline-block">مستكشف الموارد البرمجية المتقدم</h1> <button onClick={toggleDarkMode} className="ml-4 p-2 rounded-lg"> {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />} </button> <p className="mt-2 opacity-80">اكتشف أفضل المنصات والأدوات للمطورين مع ممارسات عملية مولدة بالذكاء الاصطناعي</p> </div>

{/* Controls */}
    <div className={`${card} rounded-2xl shadow-lg p-6 mb-8`}>
      <div className="flex flex-col lg:flex-row gap-4 items-center">
        <div className="relative flex-1">
          <Search className="absolute right-3 top-3 w-5 h-5 opacity-50" />
          <input
            type="text"
            placeholder="ابحث في الموارد والأدوات..."
            value={searchTerm}
            onChange={e => setSearchTerm(e.target.value)}
            className={`w-full pr-10 pl-4 py-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'bg-white border-gray-300'}`}
          />
        </div>
        <div className="flex gap-4">
          <div className="relative">
            <Filter className="absolute right-3 top-3 w-5 h-5 opacity-50" />
            <select
              value={selectedCategory}
              onChange={e => setSelectedCategory(e.target.value)}
              className={`pr-10 pl-4 py-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'bg-white border-gray-300'}`}
            >
              {Object.entries(categories).map(([val, lbl]) => <option key={val} value={val}>{lbl}</option>)}
            </select>
          </div>
          <select
            value={sortBy}
            onChange={e => setSortBy(e.target.value)}
            className={`px-4 py-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'bg-white border-gray-300'}`}
          >
            <option value="name">ترتيب بالاسم</option>
            <option value="rating">ترتيب بالتقييم</option>
            <option value="popularity">ترتيب بالشعبية</option>
          </select>
        </div>
      </div>
      <div className="flex justify-center mt-4 text-sm opacity-75">
        عدد الموارد المعروضة: {filtered.length} من أصل {resources.length}
      </div>
    </div>

    {/* Grid */}
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
      {pageItems.map(r => (
        <div key={r.id} className={`${card} rounded-xl shadow-lg hover:shadow-xl transition duration-300 overflow-hidden group`}>  
          <div className="p-6 border-b ${darkMode ? 'border-gray-700' : 'border-gray-100'}">
            <div className="flex justify-between items-start mb-3">
              <div className="flex items-center gap-3">
                {r.category === 'repositories' && <GitBranch className="w-4 h-4" />}
                {r.category === 'algorithms' && <Code className="w-4 h-4" />}
                {r.category === 'sharing' && <ExternalLink className="w-4 h-4" />}
                {r.category === 'cloud-ide' && <Zap className="w-4 h-4" />}
                {r.category === 'visualization' && <Eye className="w-4 h-4" />}
                {r.category === 'challenges' && <Star className="w-4 h-4" />}
                <h3 className="text-xl font-bold">{r.name}</h3>
              </div>
              <button onClick={() => toggleFavorite(r.id)} className={`p-2 rounded-full transition ${favorites.includes(r.id) ? 'text-red-500 hover:text-red-600' : 'opacity-50 hover:text-red-500'}`}><Heart className="w-5 h-5 ${favorites.includes(r.id) ? 'fill-current' : ''}"/></button>
            </div>
            <p className="opacity-75 text-sm mb-4">{r.description}</p>
            <div className="flex justify-between items-center mb-4">
              <div className="flex gap-1">
                {[...Array(5)].map((_, i) => <Star key={i} className={`w-4 h-4 ${i < r.rating ? 'text-yellow-400 fill-current' : 'opacity-30'}`}/>)}
              </div>
              <span className={`px-2 py-1 rounded-full text-xs font-medium ${darkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-100 text-gray-700'}`}>{r.popularity}</span>
            </div>
          </div>
          <div className="p-6 pt-4">
            <div className="grid grid-cols-2 gap-2 mb-4">
              {r.features.slice(0,4).map((f, idx) => <div key={idx} className="text-xs px-2 py-1 rounded ${darkMode ? 'bg-gray-700' : 'bg-gray-50'}">{f}</div>)}
            </div>
            <div className="flex flex-wrap gap-1 mb-4">
              {r.tags.map((t, idx) => <span key={idx} className="text-xs px-2 py-1 rounded ${darkMode ? 'bg-blue-900 text-blue-400' : 'bg-blue-100 text-blue-600'}">#{t}</span>)}
            </div>
            <div className="space-y-2">
              <a href={r.url} target="_blank" rel="noopener noreferrer" className="w-full bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white font-medium py-3 rounded-lg flex items-center justify-center gap-2"><ExternalLink className="w-4 h-4"/>زيارة الموقع</a>
              <div className="flex gap-2">
                <button onClick={() => setSelectedResource(r)} className="flex-1 py-2 rounded-lg ${darkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-100 text-gray-700'}">عرض التفاصيل</button>
                <button onClick={() => generatePracticalPractices(r)} disabled={isGeneratingPractices} className="flex-1 py-2 rounded-lg flex items-center justify-center gap-2 ${darkMode ? 'bg-green-800 text-green-400' : 'bg-green-100 text-green-700'}">
                  {isGeneratingPractices ? <Loader className="animate-spin w-4 h-4"/> : <RefreshCw className="w-4 h-4"/>} ممارسات عملية
                </button>
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>

    {/* Pagination */}
    {totalPages > 1 && (
      <div className="flex justify-center items-center gap-4 mb-8">
        <button onClick={() => setCurrentPage(p => Math.max(p-1,1))} disabled={currentPage===1} className="p-2 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-white'}"><ChevronLeft className="w-5 h-5"/></button>
        <span>{currentPage} من {totalPages}</span>
        <button onClick={() => setCurrentPage(p => Math.min(p+1,totalPages))} disabled={currentPage===totalPages} className="p-2 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-white'}"><ChevronRight className="w-5 h-5"/></button>
      </div>
    )}

    {/* No Results */}
    {filtered.length === 0 && (
      <div className="text-center py-12">
        <div className="text-6xl mb-4 opacity-50">🔍</div>
        <h3 className="text-xl font-semibold mb-2">لم يتم العثور على موارد</h3>
        <p className="opacity-75">جرب كلمات بحث مختلفة أو تغيير الفئة</p>
      </div>
    )}

    {/* Details Modal */}
    {selectedResource && (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4">
        <div className={`${card} w-full max-w-xl p-6 rounded-2xl relative`}> 
          <button onClick={() => setSelectedResource(null)} className="absolute top-4 left-4 p-1 rounded-full bg-opacity-20 hover:bg-opacity-30"><X className="w-5 h-5"/></button>
          <h2 className="text-2xl font-bold mb-2">{selectedResource.name}</h2>
          <p className="mb-4 opacity-75">{selectedResource.fullDescription}</p>
          <h3 className="text-xl font-semibold mb-2">ممارسات عملية مقترحة</h3>
          {!generatedPractices[selectedResource.id] && isGeneratingPractices && (
            <div className="flex items-center gap-2"><Loader className="animate-spin w-5 h-5"/> جاري التحميل...</div>
          )}
          {generatedPractices[selectedResource.id] && (
            <ul className="space-y-3">
              {generatedPractices[selectedResource.id].map((p,i) => (
                <li key={i} className="p-4 rounded-lg border ${darkMode ? 'border-gray-700' : 'border-gray-200'}">
                  <h4 className="font-semibold mb-1">{p.title}</h4>
                  <p className="text-sm mb-1 opacity-75">{p.description}</p>
                  <p className="text-xs italic">نصيحة: {p.tip}</p>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    )}
  </div>
</div>

); };

export default DevResourcesExplorer;







