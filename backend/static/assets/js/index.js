// Hero Text Animation
function homeHeroAnimator() {
    // Create variables
    let mainWrapper = document.getElementById("hero-anim");
    let elementsWrapper = mainWrapper.querySelectorAll(".h-hero-header-highlight");
    let firstElement = elementsWrapper[0];
    let secondElement = elementsWrapper[1];
    let firstItems = firstElement.querySelectorAll(".hero-highlight-wrapper");
    let secondItems = secondElement.querySelectorAll(".hero-highlight-wrapper");
    let tl;
    let throttle;
    let progress = 0;
    // Function for setting base values
    function setBase(ele) {
        ele.forEach((e) => {
            e.style.display = "none";
            e.style.width = "0";
        });
        ele[0].style.display = "block";
        ele[0].style.width = "auto";
    }

    function animation() {
        // Set base values
        setBase(firstItems);
        setBase(secondItems);
        // Establish timeline
        tl = gsap.timeline({
            repeat: -1
        });
        // Animate first tl
        for (let item = 0; item < firstItems.length; item++) {
            if (item == 0) {
                tl.add(gsap.to(firstItems[item], {
                    width: 0,
                    duration: 0.6
                }), 1);
                tl.add(gsap.set(firstItems[item], {
                    display: "none"
                }), ">");
            } else if ((item != firstItems.length - 1) && (item != 0)) {
                tl.add(gsap.set(firstItems[item], {
                    display: "block"
                }), ">");
                tl.add(gsap.to(firstItems[item], {
                    width: "auto",
                    duration: 0.6
                }), ">");
                tl.add(gsap.to(firstItems[item], {
                    width: 0,
                    duration: 0.6
                }), ">2.6");
                tl.add(gsap.set(firstItems[item], {
                    display: "none"
                }), ">");
            } else {
                tl.add(gsap.set(firstItems[item], {
                    display: "block"
                }), ">");
                tl.add(gsap.to(firstItems[item], {
                    width: "auto",
                    duration: 0.6
                }), ">");
                tl.add(gsap.to(firstItems[item], {
                    width: 0,
                    duration: 0.6
                }), ">2.6");
                tl.add(gsap.set(firstItems[0], {
                    display: "block"
                }), ">");
                tl.add(gsap.to(firstItems[0], {
                    width: "auto",
                    duration: 0.6
                }), ">");
            }
        }
        // Animate second tl
        for (let item = 0; item < secondItems.length; item++) {
            if (item == 0) {
                tl.add(gsap.to(secondItems[item], {
                    width: 0,
                    duration: 0.6
                }), 2.6);
                tl.add(gsap.set(secondItems[item], {
                    display: "none"
                }), ">");
            } else if ((item != secondItems.length - 1) && (item != 0)) {
                tl.add(gsap.set(secondItems[item], {
                    display: "block"
                }), ">");
                tl.add(gsap.to(secondItems[item], {
                    width: "auto",
                    duration: 0.6
                }), ">");
                tl.add(gsap.to(secondItems[item], {
                    width: 0,
                    duration: 0.6
                }), ">2.6");
                tl.add(gsap.set(secondItems[item], {
                    display: "none"
                }), ">");
            } else {
                tl.add(gsap.set(secondItems[item], {
                    display: "block"
                }), ">");
                tl.add(gsap.to(secondItems[item], {
                    width: "auto",
                    duration: 0.6
                }), ">");
                tl.add(gsap.to(secondItems[item], {
                    width: 0,
                    duration: 0.6
                }), ">2.6");
                tl.add(gsap.set(secondItems[0], {
                    display: "block"
                }), ">");
                tl.add(gsap.to(secondItems[0], {
                    width: "auto",
                    duration: 0.6
                }), ">");
            }
        }
        tl.progress(progress)
    }
    animation();
    // Restart animation on window resize
    function restartTL() {
        progress = tl.progress();
        tl.kill();
        animation();
    };
    window.addEventListener("resize", function() {
        clearTimeout(throttle);
        tl.pause();
        throttle = this.setTimeout(restartTL, 500);
    });
}
homeHeroAnimator();
// Portfolio into view
function portfolioAnimation() {
    let tl = gsap.timeline({
        scrollTrigger: {
            trigger: ".swiper.swiper-portfolio",
            start: "top 60%",
        }
    });
    tl.from(".portfolio-item", {
        y: "5%",
        opacity: 0,
        stagger: {
            each: 0.3,
            from: "start"
        },
        ease: "power1.in",
        duration: 0.5
    });
}
portfolioAnimation();
// Testimonials into view
function testimonialAnimation() {
    let tl = gsap.timeline({
        scrollTrigger: {
            trigger: ".swiper.swiper-testimonials",
            start: "top 70%",
        }
    });
    tl.from(".testimonial-item", {
        y: "5%",
        opacity: 0,
        stagger: {
            each: 0.3,
            from: "start"
        },
        ease: "power1.in",
        duration: 0.5
    });
}
testimonialAnimation();
// Service into view
function serviceAnimation1() {
    let tl = gsap.timeline({
        scrollTrigger: {
            trigger: ".service-row.service-row-1",
            start: "top 70%",
        }
    });
    tl.from(".service-item.service-row-1", {
        y: "5%",
        opacity: 0,
        stagger: {
            each: 0.3,
            from: "start"
        },
        ease: "power1.in",
        duration: 0.5
    });
}
serviceAnimation1();

function serviceAnimation2() {
    let tl = gsap.timeline({
        scrollTrigger: {
            trigger: ".service-row.service-row-2",
            start: "top 70%",
        }
    });
    tl.from(".service-item.service-row-2", {
        y: "5%",
        opacity: 0,
        stagger: {
            each: 0.3,
            from: "start"
        },
        ease: "power1.in",
        duration: 0.5
    });
}
serviceAnimation2();
// Team into view
function teamAnimation() {
    let tl = gsap.timeline({
        scrollTrigger: {
            trigger: ".h-about-team",
            start: "top 70%",
        }
    });
    tl.from(".team-member", {
        y: "5%",
        opacity: 0,
        stagger: {
            each: 0.2,
            from: "start"
        },
        ease: "power1.in",
        duration: 0.4
    });
}
teamAnimation();
var mySwiper = new Swiper('.swiper-portfolio', {
    // Optional parameters
    slidesPerView: 1,
    spaceBetween: 30,
    loop: false,
    speed: 800,
    centeredSlides: false,
    lazy: true,
    navigation: {
        nextEl: '.swiper-arrow-next',
        prevEl: '.swiper-arrow-previous',
        disabledClass: 'swiper-arrow-disabled',
    },
    keyboard: {
        enabled: true,
    },
    breakpoints: {
        0: {
            /* Webflow - mobile portrait */
            slidesPerView: 1.1,
            spaceBetween: 12,
            centeredSlides: false,
        },
        478: {
            /* Webflow - mobile landscape */
            slidesPerView: 2,
            spaceBetween: 24,
        },
        767: {
            /* Webflow - tablet */
            slidesPerView: 2,
            spaceBetween: 24,
        },
        988: {
            /* Webflow - desktop */
            slidesPerView: 2,
            spaceBetween: 40,
        },
    },
})
var mySwiper = new Swiper('.swiper-testimonials', {
    // Optional parameters
    slidesPerView: 1,
    spaceBetween: 30,
    loop: false,
    speed: 800,
    centeredSlides: false,
    lazy: true,
    navigation: {
        nextEl: '.swiper-arrow-next',
        prevEl: '.swiper-arrow-previous',
        disabledClass: 'swiper-arrow-disabled',
    },
    keyboard: {
        enabled: true,
    },
    breakpoints: {
        0: {
            /* Webflow - mobile portrait */
            slidesPerView: 1.1,
            spaceBetween: 12,
            centeredSlides: false,
        },
        478: {
            /* Webflow - mobile landscape */
            slidesPerView: 2,
            spaceBetween: 24,
        },
        767: {
            /* Webflow - tablet */
            slidesPerView: 2,
            spaceBetween: 24,
        },
        988: {
            /* Webflow - desktop */
            slidesPerView: 2.5,
            spaceBetween: 40,
        },
    },
})

// Function to fetch score breakdown
async function fetchScoreBreakdown(itemIndex, query) {
    try {
        const response = await fetch(`/score_breakdown?id=${itemIndex}&query=${encodeURIComponent(query)}`);
        if (!response.ok) {
            throw new Error('Failed to fetch score breakdown');
        }
        return await response.json();
    } catch (error) {
        console.error('Error fetching score breakdown:', error);
        return null;
    }
}

// Create tooltip element for score breakdown
function createScoreBreakdownTooltip() {
    const tooltip = document.createElement('div');
    tooltip.id = 'score-breakdown-tooltip';
    tooltip.className = 'score-tooltip';
    tooltip.style.display = 'none';
    tooltip.style.position = 'absolute';
    tooltip.style.zIndex = '1000';
    tooltip.style.background = 'rgba(0, 0, 0, 0.9)';
    tooltip.style.color = '#fff';
    tooltip.style.padding = '10px 15px';
    tooltip.style.borderRadius = '5px';
    tooltip.style.maxWidth = '300px';
    tooltip.style.boxShadow = '0 2px 10px rgba(0, 0, 0, 0.2)';
    document.body.appendChild(tooltip);
    return tooltip;
}

// Format the breakdown data as HTML
function formatScoreBreakdown(data) {
    let html = `
        <h3 style="margin: 0 0 8px 0; font-size: 16px;">Why this recommendation?</h3>
        <p style="margin: 0 0 8px 0;"><strong>Overall score:</strong> ${(data.overall_score * 100).toFixed(1)}%</p>
    `;

    // Add component scores if available
    if (data.components && Object.keys(data.components).length > 0) {
        html += '<div style="margin-bottom: 8px;"><strong>Contributing factors:</strong><ul style="margin: 5px 0; padding-left: 20px;">';
        
        if (data.components.content_match) {
            html += `<li>Content similarity: ${(data.components.content_match * 100).toFixed(1)}%</li>`;
        }
        
        if (data.components.semantic_match) {
            html += `<li>Semantic similarity: ${(data.components.semantic_match * 100).toFixed(1)}%</li>`;
        }
        
        html += '</ul></div>';
    }

    // Add matching terms if available
    if (data.matching_terms && data.matching_terms.length > 0) {
        html += '<div style="margin-bottom: 8px;"><strong>Key matching terms:</strong><ul style="margin: 5px 0; padding-left: 20px;">';
        
        data.matching_terms.forEach(term => {
            html += `<li>${term.term}</li>`;
        });
        
        html += '</ul></div>';
    }

    // Add metadata factors if available
    if (data.metadata_factors) {
        if (data.metadata_factors.genre_match) {
            html += '<p style="margin: 2px 0;"><strong>✓</strong> Genre match</p>';
        }
        
        if (data.metadata_factors.title_match) {
            html += '<p style="margin: 2px 0;"><strong>✓</strong> Title match</p>';
        }
    }

    return html;
}

// Initialize tooltip and add event listeners to score elements
function initScoreBreakdown() {
    const tooltip = createScoreBreakdownTooltip();
    let currentRequest = null;
    let tooltipTimeout = null;
    
    // Add event listeners to all score elements
    document.addEventListener('mouseover', async (event) => {
        // Check if the hovered element has the 'score-value' class
        if (event.target.classList.contains('score-value')) {
            const resultElement = event.target.closest('.result-item');
            if (resultElement) {
                const itemIndex = resultElement.dataset.index;
                const searchQuery = document.getElementById('search-input').value;
                
                // Clear any existing timeout
                if (tooltipTimeout) clearTimeout(tooltipTimeout);
                
                // Set a small delay before showing the tooltip to prevent flickering
                tooltipTimeout = setTimeout(async () => {
                    // Position the tooltip near the hovered element
                    const rect = event.target.getBoundingClientRect();
                    tooltip.style.left = `${rect.right + 10}px`;
                    tooltip.style.top = `${rect.top - 10}px`;
                    
                    // Show loading indicator
                    tooltip.innerHTML = 'Loading score breakdown...';
                    tooltip.style.display = 'block';
                    
                    // Fetch the breakdown data
                    const data = await fetchScoreBreakdown(itemIndex, searchQuery);
                    if (data && !data.error) {
                        tooltip.innerHTML = formatScoreBreakdown(data);
                    } else {
                        tooltip.innerHTML = 'Unable to load score breakdown.';
                    }
                }, 300);
            }
        }
    });
    
    // Hide tooltip when mouse leaves the score element
    document.addEventListener('mouseout', (event) => {
        if (event.target.classList.contains('score-value')) {
            if (tooltipTimeout) clearTimeout(tooltipTimeout);
            tooltipTimeout = setTimeout(() => {
                tooltip.style.display = 'none';
            }, 200);
        }
    });
    
    // Keep tooltip visible when hovering over the tooltip itself
    tooltip.addEventListener('mouseover', () => {
        if (tooltipTimeout) clearTimeout(tooltipTimeout);
    });
    
    // Hide tooltip when mouse leaves the tooltip
    tooltip.addEventListener('mouseout', () => {
        tooltip.style.display = 'none';
    });
}

// Initialize the score breakdown functionality when the page loads
document.addEventListener('DOMContentLoaded', () => {
    initScoreBreakdown();
});